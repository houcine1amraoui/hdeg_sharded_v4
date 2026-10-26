"""Bounded BRE reconstruction -> frozen-target HBF development experiment."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import time
import uuid
import numpy as np
import yaml

LEVELS=('Z','S','S_tilde','g')
FAMILIES=('quiet','older_change','arriving_change')


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)


def masks_from_changes(changes,window):
    c=np.r_[0,np.cumsum(changes,dtype=np.int64)]
    any_change=c[window:]-c[:-window]>0
    arriving=changes[window-1:]
    return dict(quiet=~any_change,older_change=any_change&~arriving,arriving_change=arriving)


def select(masks,lower,upper,quotas,seed,excluded,window):
    blocked=np.zeros(len(next(iter(masks.values()))),bool)
    for i in excluded:blocked[max(0,i-window):min(len(blocked),i+window+1)]=True
    rng=np.random.default_rng(seed);rows=[];counts={}
    for family in ('arriving_change','older_change','quiet'):
        ids=np.flatnonzero(masks[family]);ids=ids[(ids>=lower)&(ids+window<upper)];counts[family]=len(ids);chosen=[]
        for i in rng.permutation(ids):
            i=int(i)
            if blocked[i]:continue
            chosen.append(i);blocked[max(0,i-window):min(len(blocked),i+window+1)]=True
            if len(chosen)==quotas[family]:break
        require(len(chosen)==quotas[family],'Insufficient disjoint '+family+' pairs; no automatic quota change')
        rows.extend(dict(index=i,family=family) for i in sorted(chosen))
    return rows,counts


def run(args,config,out,report):
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from src.data_preparation.bre_windows import BREWindowDataset
    from src.models.hdeg.bre_hierarchy import BREHierarchy
    from src.models.hdeg.bre_semantics import load_bre_semantics
    import src.models.hdeg.dbrl as base_dbrl
    import src.common.graph.graph_builder as graph_builder
    parent=json.loads(args.hierarchy_report.read_text());pr=args.hierarchy_report.parent
    require(parent['status']=='PASS' and parent['protocol']=='bre_hierarchy_compatibility_v1','Expected verified BRE hierarchy report')
    for name,h in parent['output_hashes'].items():
        require(Path(name).name==name and sha(pr/name)==h,'Parent output differs: '+name)
    checkpoint=torch.load(pr/'initialization.pt',map_location='cpu',weights_only=True)
    require(checkpoint['protocol']=='bre_hierarchy_v1' and checkpoint['optimizer_steps']==0,'Wrong initialization contract')
    for key in ('hdeg','preprocessing','seed'):require(config.get(key)==checkpoint['configuration'].get(key),'Configuration differs: '+key)
    ds=BREWindowDataset(args.input_dir,'train',config['preprocessing']['window_size'])
    require(ds.metadata_hashes==parent['input_metadata_hashes'],'Input metadata differs from verified hierarchy')
    semantics=load_bre_semantics('configs/hdeg/bre_semantics.yaml',ds.registry)
    require(semantics==checkpoint['semantics']==json.loads((pr/'resolved_semantics.json').read_text()),'Semantic mapping differs')
    settings=config.get('bre_staged_training',{})
    require(settings.get('encoder_steps',512)==512 and settings.get('hbf_steps',512)==512 and settings.get('learning_rate',1e-4)==1e-4,'Protocol fixes 512+512 steps and lr=1e-4')
    device_name=args.device or settings.get('device','cuda' if torch.cuda.is_available() else 'cpu');device=torch.device(device_name)
    require(device_name in ('cpu','cuda') and (device_name=='cpu' or torch.cuda.is_available()),'Device unavailable')
    seed=config.get('seed',42);torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(1)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=BREHierarchy(config,semantics).to(device);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    # Verify source paths by project-relative suffix, allowing a moved checkout.
    deps={}
    for old,h in parent['source_hashes'].items():
        suffix=old[old.index('/src/')+1:] if '/src/' in old else ('configs/hdeg/bre_semantics.yaml' if old.endswith('bre_semantics.yaml') else 'run_v4_bre_hierarchy.py')
        path=Path(suffix);require(sha(path)==h,'Executed source differs: '+suffix);deps[str(path)]=h
    # The prior report did not separately hash the inherited base class/graph builder.
    # Record them now, and replay all saved outputs before any updates.
    for module in (base_dbrl,graph_builder):deps[str(Path(inspect.getfile(module)))]=sha(inspect.getfile(module))
    deps[str(Path(__file__))]=sha(__file__)
    old_selection=json.loads((pr/'selection.json').read_text());old_indices=old_selection['indices']
    replay=0.
    with np.load(pr/'forward_outputs.npz',allow_pickle=False) as saved,torch.no_grad():
        for start in range(0,len(old_indices),8):
            batch=ds.get_batch(old_indices[start:start+8])
            for side in ('X','Y'):
                reps=model.encode(torch.from_numpy(batch[side]).to(device))
                for k,v in reps.items():
                    a=v.cpu().numpy();b=saved[side+'_'+k][start:start+len(a)]
                    require(np.allclose(a,b,atol=1e-6,rtol=1e-5),'Initialization replay differs: '+side+k)
                    replay=max(replay,float(np.abs(a.astype(float)-b).max()))
    print('Building train-only transition selection...',flush=True)
    w=ds.window_size;changes=np.zeros(ds.num_rows-1,bool);previous=None
    for j,e in enumerate(ds.shards):
        a=ds.load_shard(j);v=a['values'];require(np.all(a['status']==0),'This BRE experiment requires numeric-valid master training cells; status-transition policy needed otherwise')
        changes[e['start']:e['stop']-1]=np.any(v[1:]!=v[:-1],axis=1)
        if previous is not None:changes[e['start']-1]=np.any(v[0]!=previous)
        previous=v[-1].copy()
    masks=masks_from_changes(changes,w);cut=int(.8*ds.num_rows)
    train,tc=select(masks,0,cut,dict(quiet=512,older_change=256,arriving_change=256),20261026,old_indices,w)
    probe,pc=select(masks,cut+w,ds.num_rows,dict(quiet=128,older_change=64,arriving_change=64),20261027,old_indices,w)
    write(out/'selection_manifest.json',dict(update_pairs=train,probe_pairs=probe,cut_row=cut,embargo_rows=w,
        excluded_compatibility_starts=old_indices,window_size=w,update_candidate_counts=tc,probe_candidate_counts=pc,
        input_manifest_sha256=ds.manifest_sha256,interpretation='Train-period development only; 50% quiet,25% older change,25% arriving change; not population weighting.'))
    del masks,changes
    rng=np.random.default_rng(20261028);schedule=[]
    for epoch in range(4):
        order=rng.permutation(1024);schedule.extend(order[i:i+8].tolist() for i in range(0,1024,8))
    write(out/'update_schedule.json',schedule)
    # Cache only selected data on CPU; tensors moved in batches, no full-data windows.
    data={}
    for name,rows in [('update',train),('probe',probe)]:
        batches=[ds.get_batch([r['index'] for r in rows[i:i+8]]) for i in range(0,len(rows),8)]
        data[name]={k:torch.from_numpy(np.concatenate([b[k] for b in batches])) for k in ('X','Y','X_values','Y_values','X_status','Y_status')}
        del batches
    n=model.num_devices;d=model.embedding_dim;k=model.num_states
    class Decoders(torch.nn.Module):
        def __init__(self):
            super().__init__();self.heads=torch.nn.ModuleDict({key:torch.nn.Sequential(torch.nn.Flatten(1),torch.nn.Linear(size,64),torch.nn.GELU(),torch.nn.Linear(64,w*n)) for key,size in zip(LEVELS,[n*d,k*d,k*d,d])})
        def forward(self,reps):return {key:self.heads[key](reps[key]).reshape(-1,w,n) for key in LEVELS}
    heads=Decoders().to(device)
    initial_heads={key:v.detach().cpu().clone() for key,v in heads.state_dict().items()}
    def batch(name,ids):return {key:v[ids].to(device) for key,v in data[name].items()}
    # Training-only mean reconstruction baseline; all selected cells were checked valid.
    mean_values=data['update']['X_values'].double().mean((0,1)).float().to(device)
    write(out/'value_mean_baseline.json',dict(fit_pairs=1024,fit_endpoint='X',means=mean_values.cpu().tolist()))
    original={key:v.detach().cpu().clone() for key,v in model.state_dict().items()}
    def snapshot():return {key:v.detach().cpu().clone() for key,v in model.state_dict().items()}
    def delta_norm(before,prefix):
        return float(sum((v.detach().cpu().double()-before[key].double()).square().sum() for key,v in model.named_parameters() if key.startswith(prefix)).sqrt())
    frozen_refs={};cached={};evaluations={}
    def evaluate(tag):
        model.eval();heads.eval();results={};cache={}
        with torch.no_grad():
            for name,rows in [('update',train),('probe',probe)]:
                rec=[];arrays={side:{key:[] for key in LEVELS} for side in ('X','Y')}
                for start in range(0,len(rows),8):
                    b=batch(name,slice(start,start+8));x=model.encode(b['X']);y=model.encode(b['Y']);recon=heads(x)
                    for j,row in enumerate(rows[start:start+8]):
                        rec.append(dict(**row,mean_reconstruction_mse=float((mean_values-b['X_values'][j]).square().mean()),
                            levels={key:dict(reconstruction_mse=float((recon[key][j]-b['X_values'][j]).square().mean())) for key in LEVELS}))
                    for side,rep in [('X',x),('Y',y)]:
                        for key,v in rep.items():arrays[side][key].append(v.cpu())
                arrays={side:{key:torch.cat(v) for key,v in levels.items()} for side,levels in arrays.items()};cache[name]=arrays
                summary={}
                for key in LEVELS:
                    a=arrays['X'][key].double().flatten(1);target=arrays['Y'][key].double()
                    summary[key]=dict(mean_feature_variance=float(a.var(0,unbiased=False).mean()),mean_squared_norm=float(a.square().sum(1).mean()),
                        target_drift_from_initial_mse=0. if tag=='initial' else float((target-frozen_refs[name][key]).square().mean()))
                results[name]=dict(records=rec,variation=summary)
                np.savez_compressed(out/(name+'_'+tag+'_representations.npz'),**{side+'_'+key:v.numpy() for side,levels in arrays.items() for key,v in levels.items()})
                if tag=='initial':frozen_refs[name]={key:v.double() for key,v in arrays['Y'].items()}
        write(out/('reconstruction_'+tag+'.json'),results);evaluations[tag]={name:v['variation'] for name,v in results.items()}
        return cache
    evaluate('initial')
    # Stage 1: only reconstruction learns the encoder. HBF is excluded and frozen.
    for p in model.hbf.parameters():p.requires_grad_(False)
    encoder_params=[p for name,p in model.named_parameters() if not name.startswith('hbf.')]
    parameters=encoder_params+list(heads.parameters());optimizer=torch.optim.Adam(parameters,lr=1e-4)
    history=[];t0=time.monotonic()
    for step,ids in enumerate(schedule,1):
        model.train();heads.train();optimizer.zero_grad(set_to_none=True);b=batch('update',ids)
        recon=heads(model.encode(b['X']));losses={key:(v-b['X_values']).square().mean() for key,v in recon.items()};loss=sum(losses.values())/4
        require(torch.isfinite(loss).item(),'Nonfinite reconstruction loss');loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        report['encoder_steps']=step;history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),levels={key:float(v.detach()) for key,v in losses.items()}))
        if step==1 or step%64==0:print(f'Encoder {step}/512 loss={float(loss.detach()):.6g} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    write(out/'encoder_history.json',history)
    for key,v in model.hbf.state_dict().items():require(torch.equal(v.cpu(),original['hbf.'+key]),'HBF changed during encoder training')
    cached=evaluate('encoder_final');frozen=snapshot()
    torch.save(dict(protocol='bre_reconstruction_stage_v1',model=frozen,decoders={key:v.detach().cpu() for key,v in heads.state_dict().items()},configuration=config,semantics=semantics,
        selection_sha256=sha(out/'selection_manifest.json'),encoder_steps=512,hbf_steps=0),out/'encoder_checkpoint.pt')
    for p in model.parameters():p.requires_grad_(False)
    for p in heads.parameters():p.requires_grad_(False)
    for p in model.hbf.parameters():p.requires_grad_(True)
    # Freeze cached targets; each scale/mean increment is fitted only on update pairs.
    mean={};scale={}
    for key in LEVELS:
        inc=cached['update']['Y'][key]-cached['update']['X'][key]
        mean[key]=inc.mean(0).to(device);scale[key]=inc.double().square().mean().clamp_min(1e-12).float().to(device)
    write(out/'latent_baselines.json',dict(fit_pairs=1024,scale_floor=1e-12,update_persistence_scales={key:float(v) for key,v in scale.items()}))
    np.savez_compressed(out/'mean_increments.npz',**{key:v.cpu().numpy() for key,v in mean.items()})
    summaries={}
    def forecast_eval(step):
        model.eval();records={};summary={}
        with torch.no_grad():
            for name,rows in [('update',train),('probe',probe)]:
                entries=[]
                for start in range(0,len(rows),8):
                    x={key:v[start:start+8].to(device) for key,v in cached[name]['X'].items()};y={key:v[start:start+8].to(device) for key,v in cached[name]['Y'].items()}
                    pred=model.predict(x)
                    for j,row in enumerate(rows[start:start+8]):
                        levels={}
                        for key in LEVELS:
                            def mse(v):return float(v.double().square().mean())
                            levels[key]=dict(hbf_mse=mse(pred[key][j]-y[key][j]),persistence_mse=mse(x[key][j]-y[key][j]),
                                mean_increment_mse=mse(x[key][j]+mean[key]-y[key][j]),correction_mse=mse(pred[key][j]-x[key][j]))
                            if step==0:require(torch.equal(pred[key][j],x[key][j]),'HBF stage zero is not persistence')
                        entries.append(dict(**row,levels=levels))
                records[name]=entries;summary[name]={}
                for family in ('all',)+FAMILIES:
                    group=[r for r in entries if family=='all' or r['family']==family]
                    summary[name][family]={key:{metric:float(np.mean([r['levels'][key][metric] for r in group])) for metric in ('hbf_mse','persistence_mse','mean_increment_mse','correction_mse')} for key in LEVELS}
        write(out/f'forecast_step_{step:04d}.json',dict(records=records,summary=summary));summaries[str(step)]=summary
    forecast_eval(0);optimizer=torch.optim.Adam(model.hbf.parameters(),lr=1e-4);history=[]
    for step,ids in enumerate(schedule,1):
        model.hbf.train();optimizer.zero_grad(set_to_none=True)
        x={key:v[ids].to(device) for key,v in cached['update']['X'].items()};y={key:v[ids].to(device) for key,v in cached['update']['Y'].items()}
        pred=model.predict(x);losses={key:(pred[key]-y[key]).square().mean()/scale[key] for key in LEVELS};loss=sum(losses.values())/4
        require(torch.isfinite(loss).item(),'Nonfinite forecasting loss');loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.hbf.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        report['hbf_steps']=step;history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),levels={key:float(v.detach()) for key,v in losses.items()}))
        if step==1 or step%64==0:print(f'HBF {step}/512 loss={float(loss.detach()):.6g}',flush=True)
    forecast_eval(512);write(out/'hbf_history.json',history)
    for key,v in model.state_dict().items():
        require(torch.isfinite(v).all().item(),'Nonfinite final state')
        if not key.startswith('hbf.'):require(torch.equal(v.cpu(),frozen[key]),'Encoder changed during HBF training')
    ds.verify_unchanged();require(all(sha(path)==h for path,h in deps.items()),'Source changed during experiment')
    torch.save(dict(protocol='bre_staged_training_v1',model=snapshot(),decoders={key:v.detach().cpu() for key,v in heads.state_dict().items()},
        configuration=config,semantics=semantics,selection_sha256=sha(out/'selection_manifest.json'),input_manifest_sha256=ds.manifest_sha256,
        encoder_steps=512,hbf_steps=512),out/'final_checkpoint.pt')
    report.update(status='COMPLETE',device=device_name,source_hashes=deps,parent_report_sha256=sha(args.hierarchy_report),
        initialization_sha256=sha(pr/'initialization.pt'),initialization_replay_max_abs=replay,input_manifest_sha256=ds.manifest_sha256,
        decoder_parameter_delta_l2=float(sum((v.detach().cpu().double()-initial_heads[key].double()).square().sum() for key,v in heads.named_parameters()).sqrt()),
        encoder_frozen_during_hbf=True,hbf_target_drift_mse=0.,representation_variation=evaluations,forecast_summary=summaries,
        parameter_delta_l2={name:delta_norm(original,name+'.') for name in ('dbrl','bse','bil','ebrl','hbf')},
        versions=dict(torch=torch.__version__,torch_geometric=torch_geometric.__version__,numpy=np.__version__),
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir');p.add_argument('--input_dir',type=Path,required=True)
    p.add_argument('--hierarchy_report',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda']);args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    config=copy.deepcopy(config);config['preprocessing']['dataset_name']='BRE';config['preprocessing']['merge_bre_cu']=False
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'bre_staged_training'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',protocol='bre_staged_training_v1',encoder_steps=0,hbf_steps=0,scientific_status='DEVELOPMENT_ONLY',
        contract=dict(update_pairs=1024,probe_pairs=256,batch_size=8,encoder_steps=512,hbf_steps=512,learning_rate=1e-4,clip_norm=1.,
            reconstruction='Equal mean of four normalized-value window MSEs; all cells numeric-valid in master.',
            hbf='Equal mean of four latent MSEs divided by update-only persistence MSE (floor 1e-12).'),
        limitations=['Probes are chronologically separate train-period development pairs; scaler was fitted on the full training period.',
            'Fixed quota sampling does not estimate natural population error.','BRE master imputation status cannot be recovered.',
            'BIL single-neighbor topology cannot learn relative attention weighting.','No behavior labels, actor-test values or final scientific validation.'])
    write(out/'configuration_snapshot.json',config)
    try:run(args,config,out,report)
    except Exception as e:report.update(status='BLOCKED',error=type(e).__name__+': '+str(e))
    write(out/'training_report.json',report);print('BRE staged training:',report['status']);print('Report:',out/'training_report.json')
    if report['status']!='COMPLETE':print(report['error']);raise SystemExit(2)

if __name__=='__main__':main()
