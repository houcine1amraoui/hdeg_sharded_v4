"""512 HBF-only updates on cached, frozen v4 representations; no data reselection."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import uuid
import numpy as np
import yaml

LEVELS = ('Z', 'S', 'S_tilde', 'g')
FAMILIES = ('quiet', 'older_value_change', 'arriving_value_change', 'status_change')
HEADS = ('device_prediction_head', 'state_prediction_head', 'context_prediction_head', 'global_prediction_head')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def validate_selection(selection, schedule, num_rows):
    train, probe = selection['update_pairs'], selection['probe_pairs']
    require(len(train) == 1024 and len(probe) == 256, 'Selection counts differ')
    cut = selection['cut_row']
    for rows, quota, lower, upper in ((train, 256, 0, cut), (probe, 64, cut+30, num_rows)):
        require(all(sum(r['family'] == f for r in rows) == quota for f in FAMILIES), 'Family quotas differ')
        ids = sorted(r['index'] for r in rows)
        require(all(type(i) is int and lower <= i and i+31 <= upper for i in ids), 'Partition bounds differ')
        require(all(b-a >= 31 for a, b in zip(ids, ids[1:])), 'Overlapping pairs')
        for i in ids:
            require(all(i+31 <= a or i >= b for a,b in selection['prior_diagnostic_supports_excluded']), 'Old diagnostic overlap')
    require(len(schedule) == 512 and all(len(b) == 8 for b in schedule), 'Expected 512 batches of eight')
    for epoch in range(4):
        require(sorted(i for b in schedule[epoch*128:(epoch+1)*128] for i in b) == list(range(1024)), 'Schedule is not four complete passes')


def run(args, config, out, report):
    import torch
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
    parent = json.loads(args.training_report.read_text()); root = args.training_report.parent
    require(parent['status'] == 'COMPLETE' and parent['protocol'] == 'grounded_hdeg_training_v1' and parent['optimizer_steps'] == 512, 'Expected completed 512-update grounded training')
    for name, digest in parent['output_hashes'].items():
        require(Path(name).name == name and sha(root/name) == digest, 'Parent artifact differs: '+name)
    for key in ('hdeg','preprocessing','seed'):
        require(config.get(key) == parent['configuration'].get(key), 'Configuration differs: '+key)
    deps = dict(window_loader=Path(windows.__file__),semantic_loader=Path(inspect.getfile(load_behavioral_state_config)),
                semantic_config=Path(config['project_root_dir'])/'configs/hdeg/behavioral_states.yaml')
    for cls in (ResidualContextDBRL,DBRL,GraphReasoner,GraphBuilder,BehavioralStateEstimator,BehavioralInteractionLearner,EcosystemBehavioralRepresentationLearner,HierarchicalBehavioralForecaster):
        deps[cls.__name__] = Path(inspect.getfile(cls))
    for key,path in deps.items():
        require(sha(path) == parent['source_hashes'][key], 'Executed source differs: '+key)
    ds = windows.InputStateWindowDataset(args.input_dir,30)
    require(ds.policy_version == windows.SOURCE_POLICY and ds.manifest_sha256 == parent['input_manifest_sha256'], 'Input lineage differs')
    require(ds.devices == json.loads((root/'devices.json').read_text()), 'Device order differs')
    selection = json.loads((root/'selection_manifest.json').read_text())
    schedule = json.loads((root/'update_schedule.json').read_text())
    validate_selection(selection,schedule,ds.num_rows)
    require(selection['input_manifest_sha256'] == ds.manifest_sha256, 'Selection input differs')
    settings = config.get('v4_fixed_target_hbf',{})
    device_name = args.device or settings.get('device','cuda' if torch.cuda.is_available() else 'cpu')
    require(device_name in ('cpu','cuda') and (device_name == 'cpu' or torch.cuda.is_available()), 'Device unavailable')
    require(settings.get('learning_rate',1e-4) == 1e-4 and settings.get('steps',512) == 512, 'Protocol fixes lr=1e-4 and steps=512')
    torch.set_num_threads(1);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device(device_name);mp=parent['model_parameters']
    model=torch.nn.ModuleDict(dict(dbrl=ResidualContextDBRL(**mp['dbrl']),
        bse=BehavioralStateEstimator(9,64,**mp['bse']),bil=BehavioralInteractionLearner(64,num_states=9,**mp['bil']),
        ebrl=EcosystemBehavioralRepresentationLearner(64,num_states=9),
        hbf=HierarchicalBehavioralForecaster(num_devices=23,num_states=9,embedding_dim=64,**mp['hbf'])))
    ckpt=torch.load(root/'final_checkpoint.pt',map_location='cpu',weights_only=True)
    require(ckpt['protocol']=='grounded_hdeg_v4_development' and ckpt['selection_sha256']==sha(root/'selection_manifest.json'), 'Checkpoint contract differs')
    model.load_state_dict(ckpt['model'],strict=True);model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    semantics=load_behavioral_state_config(deps['semantic_config'],root/'devices.json');mask=semantics.torch_mask(device=device)
    require(list(semantics.device_ids())==ds.devices,'Semantic device order differs')
    def encode(x):
        z=model['dbrl'](x);s=model['bse'](z,mask);st=model['bil'](s)
        return dict(Z=z,S=s,S_tilde=st,g=model['ebrl'](st))
    cached={}
    with torch.no_grad():
        for split,rows in [('update',selection['update_pairs']),('probe',selection['probe_pairs'])]:
            chunks={side:{k:[] for k in LEVELS} for side in ('X','Y')}
            for start in range(0,len(rows),8):
                b=ds.get_batch([r['index'] for r in rows[start:start+8]])
                for side in ('X','Y'):
                    for k,v in encode(torch.from_numpy(b[side]).to(device)).items():chunks[side][k].append(v.detach().cpu())
            cached[split]={s:{k:torch.cat(v).to(device) for k,v in by.items()} for s,by in chunks.items()}
            print('Cached fixed',split,'representations:',len(rows),flush=True)
    # Integrated replay: same rows and trained encoder as parent. Allow CPU/GPU reduction tolerance.
    replay={}
    for split,rows in [('update',selection['update_pairs']),('probe',selection['probe_pairs'])]:
        ids=[r['index'] for r in rows];records=json.loads((root/f'{split}_metrics_step_0512.json').read_text())['records']
        positions=[ids.index(r['index']) for r in records]
        with np.load(root/f'{split}_representations_step_0512.npz',allow_pickle=False) as saved:
            for side in ('X','Y'):
                for k in LEVELS:
                    a=cached[split][side][k][positions].cpu().numpy();b=saved[k if side=='X' else 'target_'+k]
                    require(np.allclose(a,b,rtol=1e-5,atol=1e-6), 'Parent encoder replay failed: '+split+side+k)
                    replay[split+'_'+side+'_'+k]=float(np.abs(a.astype(float)-b).max())
    # All normalization and baselines use update pairs only. No probe-dependent choices.
    delta={k:cached['update']['Y'][k]-cached['update']['X'][k] for k in LEVELS}
    scale={k:delta[k].double().square().mean().clamp_min(1e-12).float() for k in LEVELS}
    mean={k:delta[k].mean(0) for k in LEVELS}
    write(out/'training_baselines.json',dict(increment_mse_scale={k:float(v) for k,v in scale.items()},scale_floor=1e-12,fit_pairs=1024))
    np.savez_compressed(out/'mean_increments.npz',**{k:v.cpu().numpy() for k,v in mean.items()})
    # Retain learned HBF backbone; reset only its four correction heads, for exact persistence.
    parent_hbf={k:v.detach().cpu().clone() for k,v in model['hbf'].state_dict().items()}
    for name in HEADS:
        head=getattr(model['hbf'],name);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    for p in model['hbf'].parameters():p.requires_grad_(True)
    initial={k:v.detach().cpu().clone() for k,v in model['hbf'].state_dict().items()}
    def evaluate(step):
        model.eval();all_records={}
        with torch.no_grad():
            for split,rows in [('update',selection['update_pairs']),('probe',selection['probe_pairs'])]:
                records=[]
                for start in range(0,len(rows),8):
                    x={k:cached[split]['X'][k][start:start+8] for k in LEVELS};y={k:cached[split]['Y'][k][start:start+8] for k in LEVELS}
                    correction=model['hbf'](*(x[k] for k in LEVELS))
                    for j,row in enumerate(rows[start:start+8]):
                        levels={}
                        for k in LEVELS:
                            def mse(a):return float(a.double().square().mean())
                            levels[k]=dict(hbf_mse=mse(x[k][j]+correction[k][j]-y[k][j]),persistence_mse=mse(x[k][j]-y[k][j]),
                                mean_increment_mse=mse(x[k][j]+mean[k]-y[k][j]),correction_mse=mse(correction[k][j]))
                            if step==0:require(torch.count_nonzero(correction[k][j]).item()==0,'Step-zero persistence failed')
                        records.append(dict(**row,levels=levels))
                all_records[split]=records
        summary={split:{family:{k:{metric:float(np.mean([r['levels'][k][metric] for r in rows if family=='all' or r['family']==family]))
            for metric in ('hbf_mse','persistence_mse','mean_increment_mse','correction_mse')} for k in LEVELS}
            for family in ('all',)+FAMILIES} for split,rows in all_records.items()}
        write(out/('metrics_parent.json' if step=='parent' else f'metrics_step_{step:04d}.json'),dict(records=all_records,summary=summary))
        return summary
    model['hbf'].load_state_dict(parent_hbf,strict=True)
    parent_summary=evaluate('parent')
    model['hbf'].load_state_dict(initial,strict=True)
    baseline=evaluate(0);optimizer=torch.optim.Adam(model['hbf'].parameters(),lr=1e-4);history=[]
    for step,positions in enumerate(schedule,1):
        model['hbf'].train();optimizer.zero_grad(set_to_none=True)
        x={k:cached['update']['X'][k][positions] for k in LEVELS};y={k:cached['update']['Y'][k][positions] for k in LEVELS}
        correction=model['hbf'](*(x[k] for k in LEVELS))
        losses={k:(x[k]+correction[k]-y[k]).square().mean()/scale[k] for k in LEVELS};loss=sum(losses.values())/4
        require(torch.isfinite(loss).item(),'Nonfinite loss');loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model['hbf'].parameters(),1.,error_if_nonfinite=True);optimizer.step()
        report['optimizer_steps']=step
        history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),levels={k:float(v.detach()) for k,v in losses.items()}))
        if step==1 or step%64==0:print(f'HBF update {step}/512 loss={float(loss.detach()):.6g}',flush=True)
    final=evaluate(512)
    for name,value in model.state_dict().items():
        require(torch.isfinite(value).all().item(),'Nonfinite parameter')
        if not name.startswith('hbf.'):
            require(torch.equal(value.cpu(),ckpt['model'][name]),'Frozen encoder changed: '+name)
    parameter_delta=float(torch.cat([(v.detach().cpu().double()-initial[k].double()).flatten() for k,v in model['hbf'].named_parameters()]).norm())
    ds.verify_unchanged()
    for key,path in deps.items():require(sha(path)==parent['source_hashes'][key],'Source changed during execution: '+key)
    write(out/'training_history.json',history)
    write(out/'selection_manifest.json',selection);write(out/'update_schedule.json',schedule)
    torch.save(dict(protocol='fixed_target_hbf_v1',model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        observation_heads=ckpt['observation_heads'],configuration=config,parent_checkpoint_sha256=sha(root/'final_checkpoint.pt'),
        selection_sha256=sha(root/'selection_manifest.json'),optimizer_steps=512),out/'final_checkpoint.pt')
    report.update(status='COMPLETE',device=device_name,encoder_unchanged=True,target_drift_mse=0.,hbf_parameter_delta_l2=parameter_delta,
        replay_max_abs_errors=replay,parent_summary=parent_summary,baseline_summary=baseline,final_summary=final,
        parent_report_sha256=sha(args.training_report),parent_checkpoint_sha256=sha(root/'final_checkpoint.pt'),
        runner_sha256=sha(__file__),configuration=config,source_hashes={k:sha(p) for k,p in deps.items()},
        versions=dict(torch=torch.__version__,numpy=np.__version__),
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir');p.add_argument('--input_dir',type=Path,required=True)
    p.add_argument('--training_report',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda']);args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'fixed_target_hbf_training'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',protocol='fixed_target_hbf_training_v1',optimizer_steps=0,scientific_status='DEVELOPMENT_ONLY',
        objective='Mean of four latent MSEs divided by fixed update-only persistence MSEs (floor 1e-12)',
        contract=dict(steps=512,batch_size=8,learning_rate=1e-4,clip_norm=1.,seed=42,encoder_frozen=True,decoder_frozen=True,
            hbf_backbone='parent trained backbone',hbf_heads='zero reset',probe_used_for_updates=False),
        limitations=['Same explored train-period probes; not independent validation.','No behavior-discrimination or anomaly-detection claim.',
            'No observation forecasting loss; saved observation heads are retained without updates.','GPU reductions may differ across platforms.'])
    try:run(args,config,out,report)
    except Exception as error:
        report.update(status='BLOCKED',error=type(error).__name__+': '+str(error))
    write(out/'training_report.json',report)
    print('Fixed-target HBF:',report['status']);print('Report:',out/'training_report.json')
    if report['status']!='COMPLETE':print(report['error']);raise SystemExit(2)

if __name__=='__main__':main()
