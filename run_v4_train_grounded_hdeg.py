"""Bounded forward-only DBRL -> BSE -> BIL -> EBRL -> standard HBF check."""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import uuid
import numpy as np
import yaml

LEVELS=('Z','S','S_tilde','g')
STATE_IDS=('access_entry','occupancy_presence','physical_disturbance','thermal_environment',
 'switch_controlled_appliance_activity','lighting_activity','air_quality','water_heating','humidification')
EDGES=((0,1),(1,4),(1,5),(1,7),(4,3),(7,3),(8,6))
STRATA=('all_missing','partial_quiet','valid_value_transition','status_transition')


def require(c,msg):
    if not c:raise ValueError(msg)


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def state_digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name,str(a.dtype),list(a.shape)],separators=(',',':')).encode());h.update(a.tobytes())
    return h.hexdigest()


def variation(a):
    a=a.astype(np.float64).reshape(len(a),-1)
    return dict(samples=len(a),mean_feature_variance=float(a.var(axis=0).mean()),
                mean_squared_norm=float(np.square(a).sum(axis=1).mean()),
                maximum_absolute_value=float(np.abs(a).max()))


FAMILIES=('quiet','older_value_change','arriving_value_change','status_change')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def census_masks(values,status):
    def roll(a,w):
        c=np.concatenate(([0],np.cumsum(a,dtype=np.int64)));return c[w:]-c[:-w]
    sc=np.any(status[1:]!=status[:-1],axis=1)
    vc=np.any((status[1:]==0)&(status[:-1]==0)&(values[1:]!=values[:-1]),axis=1)
    any_sc=roll(sc,30)>0;any_vc=roll(vc,30)>0;arriving=vc[29:]
    # At least one currently valid device in both X and Y endpoints.
    observed=np.any(status[29:-1]==0,axis=1)&np.any(status[30:]==0,axis=1)
    return dict(quiet=observed&~any_sc&~any_vc,
        older_value_change=observed&~any_sc&any_vc&~arriving,
        arriving_value_change=observed&~any_sc&arriving,
        status_change=observed&any_sc)


def select_partition(masks,start,stop,per_family,seed,excluded):
    n=len(next(iter(masks.values())));blocked=np.zeros(n,bool)
    # A pair uses rows i..i+30 inclusive; block all intersecting starts.
    for a,b in excluded:blocked[max(0,a-30):min(n,b)]=True
    rng=np.random.default_rng(seed);chosen=[];counts={}
    for family in ('status_change','arriving_value_change','older_value_change','quiet'):
        candidates=np.flatnonzero(masks[family]);candidates=candidates[(candidates>=start)&(candidates+31<=stop)]
        counts[family]=len(candidates);take=[]
        for i in rng.permutation(candidates):
            i=int(i)
            if blocked[i]:continue
            take.append(i);blocked[max(0,i-30):min(n,i+31)]=True
            if len(take)==per_family:break
        require(len(take)==per_family,'Insufficient nonoverlapping '+family+' pairs; no automatic quota relaxation')
        chosen.extend(dict(index=i,family=family) for i in sorted(take))
    return chosen,counts


def run(args,config,out,report):
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from src.data_preparation import input_state_windows as windows
    from src.models.hdeg.residual_context_dbrl import ResidualContextDBRL
    from src.models.hdeg.dbrl import DBRL
    from src.models.hdeg.graph_reasoner import GraphReasoner
    from src.common.graph.graph_builder import GraphBuilder
    from src.models.hdeg.bse import BehavioralStateEstimator
    from src.models.hdeg.bil import BehavioralInteractionLearner
    from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner
    from src.models.hdeg.hbf import HierarchicalBehavioralForecaster
    from src.common.graph.semantics import load_behavioral_state_config
    proof=json.loads(args.residual_report.read_text());hierarchy=json.loads(args.hierarchy_report.read_text())
    require(proof['status']=='COMPLETE' and proof['protocol']=='source_residual_context_check_v1','Residual check not complete')
    require(hierarchy['status']=='PASS' and sha(args.hierarchy_report)==proof['source_hashes']['hierarchy_report'],'Hierarchy lineage differs')
    for k in ('hdeg','preprocessing','seed'):
        require(config.get(k)==hierarchy['configuration'].get(k),'Reference configuration differs: '+k)
    settings=config.get('v4_grounded_training',{})
    steps=args.steps if args.steps is not None else settings.get('steps',512)
    lr=settings.get('learning_rate',1e-4);batch_size=8
    require(type(steps) is int and 1<=steps<=512,'Steps must be 1..512')
    require(lr==1e-4,'This frozen development protocol uses learning_rate=1e-4')
    device_name=args.device or settings.get('device','cuda' if torch.cuda.is_available() else 'cpu')
    require(device_name in ('cpu','cuda') and (device_name=='cpu' or torch.cuda.is_available()),'Device unavailable')
    deps=dict(residual_report=args.residual_report,hierarchy_report=args.hierarchy_report,
        initialization=args.hierarchy_report.parent/'initialization.pt',
        residual_arrays=args.residual_report.parent/'residual_context_outputs.npz',
        frozen_selection=args.residual_report.parent/'consumed_selection.json',
        config=Path('configs/config.yaml'),runner=Path(__file__),window_loader=Path(windows.__file__),
        semantic_config=Path(config['project_root_dir'])/'configs/hdeg/behavioral_states.yaml',
        semantic_loader=Path(inspect.getfile(load_behavioral_state_config)))
    for cls in [ResidualContextDBRL,DBRL,GraphReasoner,GraphBuilder,BehavioralStateEstimator,BehavioralInteractionLearner,
                EcosystemBehavioralRepresentationLearner,HierarchicalBehavioralForecaster]:deps[cls.__name__]=Path(inspect.getfile(cls))
    hashes={k:sha(v) for k,v in deps.items()}
    require(hashes['initialization']==proof['initialization_file_sha256']==hierarchy['initialization_file_sha256'],'Initialization differs')
    for k in ('residual_arrays','frozen_selection'):
        require(hashes[k]==proof['output_hashes'][deps[k].name],'Residual output differs')
    for k in hierarchy['source_hashes']:
        if k in deps and k not in ('runner','config'):
            require(hashes[k]==hierarchy['source_hashes'][k],'Source differs from hierarchy: '+k)
    ds=windows.InputStateWindowDataset(args.input_dir,30)
    require(ds.policy_version==windows.SOURCE_POLICY and ds.manifest_sha256==proof['input_manifest_sha256']==hierarchy['manifest_sha256'],'Input lineage differs')
    require(ds.metadata_hashes==proof['metadata_hashes'],'Metadata differs')
    require(ds.num_rows<=3_000_000,'Input census bound exceeded')
    write(out/'devices.json',ds.devices)
    semantic=load_behavioral_state_config(deps['semantic_config'],out/'devices.json')
    require(list(semantic.device_ids())==ds.devices and list(semantic.state_ids())==hierarchy['state_ids'],'Semantic order differs')
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    # CPU replay is deterministic; GPU training may have nondeterministic graph reductions.
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    mp=hierarchy['model_parameters'];d=64
    model=torch.nn.ModuleDict(dict(dbrl=ResidualContextDBRL(**mp['dbrl']),
        bse=BehavioralStateEstimator(9,d,**mp['bse']),
        bil=BehavioralInteractionLearner(d,num_states=9,**mp['bil']),
        ebrl=EcosystemBehavioralRepresentationLearner(d,num_states=9),
        hbf=HierarchicalBehavioralForecaster(num_devices=23,num_states=9,embedding_dim=d,**mp['hbf'])))
    model.load_state_dict(torch.load(deps['initialization'],map_location='cpu',weights_only=True),strict=True)
    require(state_digest(model)==hierarchy['initialization_state_sha256'],'Loaded hierarchy state differs')
    model.eval()
    # The CPU compatibility replay is integrated, not a separate user task.
    selection=json.loads(deps['frozen_selection'].read_text());old_indices=[]
    for event in selection['selected_events']:old_indices.extend(p['index'] for p in event['placements'])
    old_indices.extend(c['index'] for c in selection['controls'])
    replay_error=0.
    with np.load(deps['residual_arrays'],allow_pickle=False) as reference,torch.inference_mode():
        for start in range(0,60,8):
            batch=ds.get_batch(old_indices[start:start+8])
            for b in ('X','Y'):
                actual=model['dbrl'](torch.from_numpy(batch[b])).numpy();expected=reference[b+'_R'][start:start+len(actual)]
                replay_error=max(replay_error,float(np.abs(actual.astype(float)-expected.astype(float)).max()))
                require(np.array_equal(actual,expected),'Exact full-DBRL residual replay failed')
    # Deliberate HBF methodology change: residual prediction initialized at persistence.
    for name in ('device_prediction_head','state_prediction_head','context_prediction_head','global_prediction_head'):
        head=getattr(model['hbf'],name);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    class ObservationHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads=torch.nn.ModuleDict({k:torch.nn.Sequential(torch.nn.Flatten(1),torch.nn.Linear(size,64),
                torch.nn.GELU(),torch.nn.Linear(64,30*23*6)) for k,size in zip(LEVELS,[23*64,9*64,9*64,64])})
        def forward(self,reps):return {k:self.heads[k](reps[k]).reshape(-1,30,23,6) for k in LEVELS}
    heads=ObservationHeads();device=torch.device(device_name)
    model.to(device);heads.to(device);mask=semantic.torch_mask(device=device)
    require(mask.cpu().tolist()==hierarchy['compatibility_mask'],'Semantic mask differs')
    require(model['bil'].edge_index.cpu().tolist()==np.array(hierarchy['big_edges']).T.tolist(),'BIL graph differs')
    def encode(x):
        z=model['dbrl'](x);s=model['bse'](z,mask);st=model['bil'](s)
        return dict(Z=z,S=s,S_tilde=st,g=model['ebrl'](st))
    def predict(current):
        delta=model['hbf'](*(current[k] for k in LEVELS))
        return {k:current[k]+delta[k] for k in LEVELS}
    print('Building fixed train-period selection...',flush=True)
    values=np.empty((ds.num_rows,23),np.float32);status=np.empty((ds.num_rows,23),np.uint8)
    for i,entry in enumerate(ds.shards):
        b=ds.load_shard(i);values[entry['start']:entry['stop']]=b['values'];status[entry['start']:entry['stop']]=b['status']
    masks=census_masks(values,status);cut=int(ds.num_rows*.8)
    excluded=[(i,i+31) for i in old_indices]
    train,tc=select_partition(masks,0,cut,256,20261027,excluded)
    probe,pc=select_partition(masks,cut+30,ds.num_rows,64,20261028,excluded)
    del values,status,masks
    selection_new=dict(protocol='grounded_training_selection_v1',input_manifest_sha256=ds.manifest_sha256,
        cut_row=cut,embargo_rows=30,update_pairs=train,probe_pairs=probe,update_candidates=tc,probe_candidates=pc,
        prior_diagnostic_supports_excluded=excluded,
        interpretation='Both partitions are inside the existing training period; probes are development data, not held-out validation.')
    write(out/'selection_manifest.json',selection_new)
    def load(rows):
        b=ds.get_batch([r['index'] for r in rows])
        return {k:torch.from_numpy(v).to(device) for k,v in b.items() if k in ('X','Y','X_values','Y_values','X_status','Y_status')}
    # Fixed training-fitted value means/status modes, only from update target windows.
    sums=np.zeros(23);counts=np.zeros(23);status_counts=np.zeros((23,5),np.int64)
    for start in range(0,len(train),8):
        b=ds.get_batch([r['index'] for r in train[start:start+8]]);v=b['Y_values'];s=b['Y_status'];valid=s==0
        sums+=(v*valid).sum(axis=(0,1));counts+=valid.sum(axis=(0,1))
        for c in range(5):status_counts[:,c]+=(s==c).sum(axis=(0,1))
    mean=torch.tensor(np.divide(sums,counts,out=np.zeros(23),where=counts>0),dtype=torch.float32,device=device)
    mode=torch.tensor(status_counts.argmax(1),dtype=torch.long,device=device)
    mean_supported=torch.tensor(counts>0,device=device)
    write(out/'training_baselines.json',dict(value_mean=mean.cpu().tolist(),valid_counts=counts.tolist(),status_mode=mode.cpu().tolist()))
    def observation_loss(decoded,values,status):
        valid=status==0;den=valid.sum().clamp_min(1)
        val=((decoded[...,0]-values)**2*valid).sum()/den
        stat=F.cross_entropy(decoded[...,1:].reshape(-1,5),status.long().reshape(-1))
        return val+0.1*stat
    rng=np.random.default_rng(20261029)
    schedule=[]
    while len(schedule)<steps:
        permutation=rng.permutation(len(train))
        schedule.extend(permutation[i:i+8].tolist() for i in range(0,len(train),8))
    schedule=schedule[:steps];write(out/'update_schedule.json',schedule)
    parameters=list(model.parameters())+list(heads.parameters());optimizer=torch.optim.Adam(parameters,lr=lr)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_heads={k:v.detach().cpu().clone() for k,v in heads.state_dict().items()}
    report.update(status='RUNNING',source_hashes=hashes,input_manifest_sha256=ds.manifest_sha256,
        replay_max_abs_error=replay_error,device=device_name,steps=steps,learning_rate=lr,
        model_parameters=mp,framework_versions=dict(torch=torch.__version__,torch_geometric=torch_geometric.__version__,numpy=np.__version__),
        objective='mean_levels(current_window_obs_reconstruction + arriving_row_obs_forecast + 0.1*stop_gradient_latent_forecast_MSE); obs=value_MSE_valid+0.1*status_CE',
        methodology_changes=['Residual eligibility-masked dense DBRL gamma=1; exact local fallback.',
            'HBF predicts increments; output heads start at zero, giving persistence at step zero.',
            'Per-level auxiliary observation heads reconstruct current and next windows.',
            'Latent targets stop gradients; observation targets remain fixed data.'],
        update_pairs=len(train),probe_pairs=len(probe),evaluation_steps=[0,steps])
    references={};evaluation_results={}
    def evaluate(name,rows,step):
        model.eval();heads.eval();records=[];representations={k:[] for k in LEVELS};targets={k:[] for k in LEVELS}
        with torch.no_grad():
            for start in range(0,len(rows),8):
                batch_rows=rows[start:start+8];b=load(batch_rows);current=encode(b['X']);target=encode(b['Y']);pred=predict(current);decoded=heads(pred)
                last_values=torch.cat((b['X_values'][:,1:,:],b['X_values'][:,-1:,:]),dim=1)
                last_status=torch.cat((b['X_status'][:,1:,:],b['X_status'][:,-1:,:]),dim=1)
                for j,row in enumerate(batch_rows):
                    valid=b['Y_status'][j]==0;stable=valid&(last_status[j]==0);entry=dict(index=row['index'],family=row['family'],levels={})
                    def mse(v,t,m):return None if not m.any() else float(((v-t)[m]**2).mean())
                    entry['baselines']=dict(persistence_valid_coverage=int(stable.sum()),valid_count=int(valid.sum()),
                        value_persistence_mse=mse(last_values[j],b['Y_values'][j],stable),
                        train_mean_mse=mse(mean.expand_as(b['Y_values'][j]),b['Y_values'][j],valid&mean_supported),
                        arriving_valid_count=int(valid[-1].sum()),arriving_persistence_valid_count=int(stable[-1].sum()),
                        arriving_train_mean_mse=mse(mean,b['Y_values'][j,-1],valid[-1]&mean_supported),
                        arriving_status_persistence_error=float((last_status[j,-1]!=b['Y_status'][j,-1]).float().mean()),
                        status_persistence_error=float((last_status[j]!=b['Y_status'][j]).float().mean()),
                        status_mode_error=float((mode.expand_as(b['Y_status'][j])!=b['Y_status'][j]).float().mean()))
                    for k in LEVELS:
                        d=decoded[k][j]
                        value_pred=torch.cat((b['X_values'][j,1:],d[-1:,:,0]),dim=0)
                        status_pred=torch.cat((b['X_status'][j,1:],d[-1:,:,1:].argmax(-1)),dim=0)
                        entry['levels'][k]=dict(
                            latent_forecast_mse=float(((pred[k][j]-target[k][j])**2).mean()),
                            latent_persistence_mse=float(((current[k][j]-target[k][j])**2).mean()),
                            value_forecast_mse=mse(value_pred,b['Y_values'][j],valid),
                            value_forecast_on_persistence_support=mse(value_pred,b['Y_values'][j],stable),
                            arriving_value_forecast_mse=mse(d[-1,:,0],b['Y_values'][j,-1],valid[-1]),
                            arriving_persistence_mse=mse(last_values[j,-1],b['Y_values'][j,-1],stable[-1]),
                            arriving_forecast_on_persistence_support=mse(d[-1,:,0],b['Y_values'][j,-1],stable[-1]),
                            status_error=float((status_pred!=b['Y_status'][j]).float().mean()),
                            arriving_status_error=float((status_pred[-1]!=b['Y_status'][j,-1]).float().mean()))
                    records.append(entry)
                for k in LEVELS:
                    representations[k].append(current[k].cpu().numpy());targets[k].append(target[k].cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in representations.items()};ys={k:np.concatenate(v) for k,v in targets.items()}
        summary={}
        for k in LEVELS:
            summary[k]=dict(variation=variation(arrays[k]),target_variation=variation(ys[k]),
                target_drift_mse=0. if step==0 else float(np.square(ys[k].astype(float)-references[name][k]).mean()))
        family_results={}
        for family in FAMILIES:
            group=[r for r in records if r['family']==family]
            def average(values):
                values=[v for v in values if v is not None]
                return dict(mean=None if not values else float(np.mean(values)),windows=len(values))
            family_results[family]={k:{metric:average([r['levels'][k][metric] for r in group]) for metric in
                ('arriving_value_forecast_mse','arriving_forecast_on_persistence_support','arriving_persistence_mse',
                 'arriving_status_error','latent_forecast_mse','latent_persistence_mse')} for k in LEVELS}
        if step==0:references[name]={k:v.astype(float) for k,v in ys.items()}
        np.savez_compressed(out/f'{name}_representations_step_{step:04d}.npz',**arrays,**{'target_'+k:v for k,v in ys.items()})
        write(out/f'{name}_metrics_step_{step:04d}.json',dict(records=records,summary=summary,family_results=family_results,
            aggregation='Equal weight per supported window within each family; persistence comparisons use identical support.'))
        evaluation_results[name+'_'+str(step)]=summary
        if name=='probe':report.setdefault('probe_family_results',{})[str(step)]=family_results
        print('Evaluated',name,'step',step,flush=True)
    # A fixed update subset provides a comparable train-side report without evaluating all 1024 pairs repeatedly.
    train_eval=[r for fam in FAMILIES for r in [v for v in train if v['family']==fam][:64]]
    evaluate('update',train_eval,0);evaluate('probe',probe,0)
    history=[];import time;t0=time.monotonic()
    for step,positions in enumerate(schedule,1):
        model.train();heads.train();b=load([train[i] for i in positions]);optimizer.zero_grad(set_to_none=True)
        current=encode(b['X'])
        with torch.no_grad():target=encode(b['Y'])
        pred=predict(current);recon=heads(current);forecast=heads(pred);parts={}
        for k in LEVELS:
            parts[k]=dict(reconstruction=observation_loss(recon[k],b['X_values'],b['X_status']),
                forecasting=observation_loss(forecast[k][:,-1:],b['Y_values'][:,-1:],b['Y_status'][:,-1:]),
                latent=((pred[k]-target[k])**2).mean())
        loss=sum(v['reconstruction']+v['forecasting']+0.1*v['latent'] for v in parts.values())/4
        require(torch.isfinite(loss).item(),'Nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        report['optimizer_steps']=step
        history.append(dict(step=step,loss=float(loss.detach()),gradient_norm_before_clip=float(norm),
            levels={k:{n:float(v.detach()) for n,v in part.items()} for k,part in parts.items()}))
        if step==1 or step%32==0 or step==steps:
            print(f'Update {step}/{steps} loss={float(loss.detach()):.6g} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    evaluate('update',train_eval,steps);evaluate('probe',probe,steps)
    updates={}
    for component in model:
        differences=[];baseline=[]
        for name,v in model[component].named_parameters():
            old=initial[component+'.'+name].double();differences.append((v.detach().cpu().double()-old).reshape(-1));baseline.append(old.reshape(-1))
        delta=torch.cat(differences);base=torch.cat(baseline)
        updates[component]=dict(delta_l2=float(delta.norm()),initial_l2=float(base.norm()),max_abs=float(delta.abs().max()))
    updates['observation_heads']=dict(delta_l2=float(torch.cat([(v.detach().cpu().double()-initial_heads[k].double()).reshape(-1) for k,v in heads.named_parameters()]).norm()))
    require(all(torch.isfinite(v).all().item() for v in parameters),'Nonfinite final parameter')
    ds.verify_unchanged();require(all(sha(path)==hashes[k] for k,path in deps.items()),'Dependency changed during run')
    write(out/'training_history.json',history)
    torch.save(dict(protocol='grounded_hdeg_v4_development',model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        observation_heads={k:v.detach().cpu() for k,v in heads.state_dict().items()},configuration=config,
        optimizer_steps=steps,selection_sha256=sha(out/'selection_manifest.json')),out/'final_checkpoint.pt')
    report.update(status='COMPLETE',parameter_updates=updates,evaluation_summary=evaluation_results,
        source_files_unchanged=True,output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description='Bounded input-grounded end-to-end HDEG development training')
    p.add_argument('--project_root_dir');p.add_argument('--input_dir',required=True,type=Path)
    p.add_argument('--residual_report',required=True,type=Path);p.add_argument('--hierarchy_report',required=True,type=Path)
    p.add_argument('--steps',type=int);p.add_argument('--device',choices=['cpu','cuda']);args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'grounded_hdeg_training'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',protocol='grounded_hdeg_training_v1',configuration=config,optimizer_steps=0,
        scientific_validation_status='DEVELOPMENT_ONLY',limitations=[
        'Probe windows are disjoint train-period development data; training-period scaling includes them.',
        'Four-family sampling is intentionally balanced and does not estimate natural-population performance.',
        'One-step windows overlap by 29 observations; arriving-row metrics must be assessed separately.',
        'Observation heads and objective are an explicit methodology revision; not the original paper protocol.',
        'Latent target drift and representation variation are measured; latent loss alone cannot establish success.',
        'No activity labels, behavior-discrimination validation, test-set access or automatic manuscript claims.',
        'GPU graph reductions may be nondeterministic; source/seed records do not guarantee bitwise GPU training.'])
    try:run(args,config,out,report)
    except Exception as exc:report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write(out/'training_report.json',report);print('Grounded HDEG training:',report['status']);print('Report:',out/'training_report.json')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2
if __name__=='__main__':raise SystemExit(main())
