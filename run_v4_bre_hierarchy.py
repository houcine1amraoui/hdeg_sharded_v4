"""Integrated train-only BRE loader/hierarchy check; no optimizer or training."""
import argparse
import copy
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path
import uuid
import numpy as np
import yaml
from src.data_preparation.bre_windows import BREWindowDataset,require,sha
from src.models.hdeg.bre_semantics import load_bre_semantics


def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)


def run(args,config,out,report):
    import torch
    import torch_geometric
    from src.models.hdeg.bre_hierarchy import BREHierarchy,LEVELS
    ds=BREWindowDataset(args.input_dir,'train',config['preprocessing']['window_size'])
    require(config['preprocessing']==ds.configuration['preprocessing'],'Preprocessing configuration differs from BRE preparation')
    path=Path('configs/hdeg/bre_semantics.yaml');semantics=load_bre_semantics(path,ds.registry)
    source_paths={str(Path(inspect.getfile(cls)).resolve()) for cls in [BREWindowDataset,BREHierarchy,load_bre_semantics]}
    source_paths.update([str(path.resolve()),str(Path(__file__).resolve())])
    seed=config.get('seed',42);torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=BREHierarchy(config,semantics).to(device).eval()
    for module in model.modules():
        file=inspect.getfile(type(module))
        if 'src/' in file.replace('\\','/'):source_paths.add(str(Path(file).resolve()))
    hashes={p:sha(p) for p in source_paths}
    n=args.windows;require(4<=n<=64,'Windows must be 4..64')
    indices=[0,len(ds)-1]
    for e in ds.shards[:-1]:
        i=e['stop']-ds.window_size//2
        if 0<=i<len(ds):indices.append(i)
        if len(indices)>=n:break
    rng=np.random.default_rng(seed)
    while len(indices)<n:
        i=int(rng.integers(len(ds)))
        if i not in indices:indices.append(i)
    # Verify invalid requests fail, rather than wrapping or crossing the split.
    for invalid in (-1,len(ds)):
        try:ds.get_batch([invalid])
        except ValueError:pass
        else:raise ValueError('Split-boundary protection failed')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    saved={};batch_shapes=[]
    with torch.no_grad():
        for start in range(0,n,8):
            b=ds.get_batch(indices[start:start+8]);x=torch.from_numpy(b['X']).to(device);y=torch.from_numpy(b['Y']).to(device)
            require(np.array_equal(b['X'][:,1:],b['Y'][:,:-1]),'Paired windows are misaligned')
            current=model.encode(x);target=model.encode(y);pred=model.predict(current)
            isolated=~model.bil.incoming_mask.any(-1)
            require(torch.equal(current['S'][:,isolated],current['S_tilde'][:,isolated]),'Isolated BIL states changed')
            expected={'Z':(len(x),len(ds.devices),model.embedding_dim),'S':(len(x),model.num_states,model.embedding_dim),
                'S_tilde':(len(x),model.num_states,model.embedding_dim),'g':(len(x),model.embedding_dim)}
            for group,values in [('X',current),('Y',target),('forecast',pred)]:
                for k,v in values.items():
                    require(tuple(v.shape)==expected[k] and torch.isfinite(v).all().item(),'Invalid hierarchy output: '+k)
                    saved.setdefault(group+'_'+k,[]).append(v.cpu().numpy())
            for k in LEVELS:require(torch.equal(current[k],pred[k]),'Initial residual HBF is not persistence')
            batch_shapes.append(list(x.shape))
    # A single backward smoke check (no optimizer) catches invalid masked attention gradients.
    model.train();model.zero_grad(set_to_none=True)
    x=torch.from_numpy(ds.get_batch(indices[:2])['X']).to(device)
    values=model(x);loss=sum(v.square().mean() for v in values['representations'].values())+sum(v.square().mean() for v in values['predictions'].values())
    loss.backward();gradient_checks={}
    for name,module in [('dbrl',model.dbrl),('bse',model.bse),('bil',model.bil),('ebrl',model.ebrl),('hbf',model.hbf)]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        require(grads and all(torch.isfinite(g).all().item() for g in grads),'Missing or nonfinite gradients: '+name)
        gradient_checks[name]=dict(tensors=len(grads),nonzero_tensors=sum(bool(torch.count_nonzero(g).item()) for g in grads))
    model.zero_grad(set_to_none=True);model.eval()
    for k,v in model.state_dict().items():require(torch.equal(v.cpu(),initial[k]),'Compatibility check changed initialization')
    ds.verify_unchanged()
    require(all(sha(p)==h for p,h in hashes.items()),'Executed source changed')
    write(out/'resolved_semantics.json',semantics);write(out/'configuration_snapshot.json',config)
    write(out/'selection.json',dict(split='train',indices=indices,window_size=ds.window_size,input_manifest_sha256=ds.manifest_sha256,
        purpose='Compatibility only; boundary-enriched deterministic selection, not scientific sampling.'))
    np.savez_compressed(out/'forward_outputs.npz',**{k:np.concatenate(v) for k,v in saved.items()})
    torch.save(dict(protocol='bre_hierarchy_v1',model=initial,configuration=config,semantics=semantics,
        input_manifest_sha256=ds.manifest_sha256,optimizer_steps=0),out/'initialization.pt')
    report.update(status='PASS',device=device,num_devices=len(ds.devices),num_states=model.num_states,embedding_dim=model.embedding_dim,
        total_train_pairs=len(ds),tested_windows=n,input_shapes=batch_shapes,state_member_counts=semantics['member_counts'],
        gradient_checks=gradient_checks,backward_calls=1,initialization_unchanged=True,step_zero_persistence=True,
        input_metadata_hashes=ds.metadata_hashes,consumed_shard_hashes=ds.loaded_hashes,source_hashes=hashes,
        versions=dict(torch=torch.__version__,torch_geometric=torch_geometric.__version__,numpy=np.__version__),
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir');p.add_argument('--input_dir',type=Path,required=True)
    p.add_argument('--windows',type=int,default=32);p.add_argument('--device',choices=['cpu','cuda']);args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    config=copy.deepcopy(config);config['preprocessing']['dataset_name']='BRE';config['preprocessing']['merge_bre_cu']=False
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'bre_hierarchy'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',protocol='bre_hierarchy_compatibility_v1',optimizer_steps=0,scientific_status='COMPATIBILITY_ONLY',
        limitations=['Numeric validity is master-cell validity, not original observation availability.',
          'Five state groups summarize measurements, not labeled human activities.',
          'BRE interaction edges are exploratory contextual hypotheses, not established causal relations.',
          'No test-value access, model optimization or scientific performance evaluation.'])
    try:run(args,config,out,report)
    except Exception as e:report.update(status='BLOCKED',error=type(e).__name__+': '+str(e))
    write(out/'hierarchy_report.json',report);print('BRE hierarchy:',report['status']);print('Report:',out/'hierarchy_report.json')
    if report['status']!='PASS':print(report['error']);raise SystemExit(2)

if __name__=='__main__':main()
