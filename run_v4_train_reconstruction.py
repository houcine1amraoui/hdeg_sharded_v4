"""Migrate BRE encoder+decoders, continue reconstruction only, calibrate and validate."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.data_preparation.bre_windows import BREWindowDataset
from src.models.hdeg.bre_semantics import load_bre_semantics
from src.models.hdeg.bre_reconstruction import BREReconstructionModel
from src.models.hdeg.reconstruction import LEVELS, ReconstructionObjective
from src.scientific_validation.reconstruction.data import window_batch, evenly_spaced_starts, fit_threshold
from src.scientific_validation.reconstruction.runtime import (
    require, sha, write, load_config, settings, configure_device, new_output, source_hashes, check_sources, evaluate)


def run(args, config, out, report):
    opts = settings(config); device = configure_device(config,args.device)
    parent = json.loads(args.training_report.read_text()); root = args.training_report.parent
    require(parent['status']=='COMPLETE' and parent['protocol']=='bre_staged_training_v1','Expected completed BRE staged-training report')
    for name,digest in parent['output_hashes'].items():
        require(Path(name).name==name and sha(root/name)==digest,'Parent artifact differs: '+name)
    checkpoint = torch.load(root/'encoder_checkpoint.pt',map_location='cpu',weights_only=True)
    require(checkpoint['encoder_steps']==512,'Expected 512-step reconstruction encoder')
    # Ignore new reconstruction settings when comparing original architecture configuration.
    old_cfg=checkpoint['configuration']; current_hdeg={k:v for k,v in config['hdeg'].items() if k!='reconstruction'}
    require(current_hdeg=={k:v for k,v in old_cfg['hdeg'].items() if k!='reconstruction'},'Encoder configuration differs')
    require(config['preprocessing']==old_cfg['preprocessing'] and config.get('seed')==old_cfg.get('seed'),'Data/seed configuration differs')
    ds=BREWindowDataset(args.input_dir,'train',config['preprocessing']['window_size'])
    require(ds.manifest_sha256==parent['input_manifest_sha256'],'Input manifest differs')
    semantics=load_bre_semantics('configs/hdeg/bre_semantics.yaml',ds.registry)
    require(semantics==checkpoint['semantics'],'Semantic mapping differs')
    model=BREReconstructionModel(config,semantics).to(device);model.import_encoder_stage(checkpoint);model.eval()
    require(not any(k.startswith('hbf.') for k in model.state_dict()),'HBF present in reconstruction model')
    dependencies=source_hashes(model)
    # Current shared encoder files must match the parent. New files intentionally have new contracts.
    for old,digest in parent['source_hashes'].items():
        name='src/'+old.split('/src/',1)[1] if '/src/' in old else old
        if name in dependencies:require(dependencies[name]==digest,'Shared encoder source differs: '+name)
    oldsel=json.loads((root/'selection_manifest.json').read_text())
    require(checkpoint['selection_sha256']==sha(root/'selection_manifest.json'),'Checkpoint selection differs')
    updates=[r['index'] for r in oldsel['update_pairs']];cut=oldsel['cut_row'];w=ds.window_size
    require(len(updates)==1024 and all(0<=i and i+w<=cut for i in updates),'Unexpected update support')
    # Decoder migration is checked against the previously saved actual errors, not just shape compatibility.
    saved=json.loads((root/'reconstruction_encoder_final.json').read_text())['update']['records']
    from src.models.hdeg.bre_hierarchy import BREHierarchy
    reference=BREHierarchy(old_cfg,semantics).to(device);reference.load_state_dict(checkpoint['model'],strict=True);reference.eval()
    migration_error=0.;representation_replay_error=0.
    with np.load(root/'update_encoder_final_representations.npz',allow_pickle=False) as reps,torch.no_grad():
        for start in range(0,32,8):
            b=window_batch(ds,updates[start:start+8]);result=model(torch.from_numpy(b['X']).to(device))
            reference_reps=reference.encode(torch.from_numpy(b['X']).to(device))
            for k in LEVELS:
                require(torch.equal(result['representations'][k],reference_reps[k]),'Same-device migration differs')
                a=result['representations'][k].cpu().numpy()
                representation_replay_error=max(representation_replay_error,float(np.max(abs(a-reps['X_'+k][start:start+8]))))
                require(np.allclose(a,reps['X_'+k][start:start+8],atol=1e-5,rtol=1e-5),'Saved encoder replay differs beyond CPU/CUDA tolerance')
                mse=(result['reconstructions'][k]-torch.from_numpy(b['values']).to(device)).square().mean((1,2)).cpu().numpy()
                expected=np.array([r['levels'][k]['reconstruction_mse'] for r in saved[start:start+8]])
                require(np.allclose(mse,expected,atol=1e-7,rtol=1e-5),'Decoder migration replay differs')
                migration_error=max(migration_error,float(np.max(abs(mse-expected))))
    del reference
    calibration=evenly_spaced_starts(cut+w,ds.num_rows,w,opts['calibration_windows'])
    val=BREWindowDataset(args.input_dir,'val',w)
    validation=evenly_spaced_starts(0,val.num_rows,w,opts['validation_windows'])
    # Counts and values for training-only mean baseline, no validation fitting.
    sums=np.zeros(model.num_devices);counts=np.zeros(model.num_devices)
    cached=[]
    for start in range(0,len(updates),opts['batch_size']):
        b=window_batch(ds,updates[start:start+opts['batch_size']]);valid=b['status']==0
        sums+=np.where(valid,b['values'],0).sum((0,1),dtype=np.float64);counts+=valid.sum((0,1));cached.append(b)
    require(np.all(counts>0),'A channel has no valid update targets')
    mean=torch.tensor(sums/counts,dtype=torch.float32,device=device)
    data={k:torch.from_numpy(np.concatenate([b[k] for b in cached])) for k in ('X','values','status')};del cached
    rng=np.random.default_rng(config.get('seed',42)+701);schedule=[]
    while len(schedule)<opts['steps']:
        order=rng.permutation(len(updates));schedule.extend(order[i:i+opts['batch_size']].tolist() for i in range(0,len(order),opts['batch_size']))
    schedule=schedule[:opts['steps']]
    write(out/'selection.json',dict(update=updates,calibration=calibration,validation=validation,cut_row=cut,window_size=w,
        calibration_rule='Evenly spaced nonoverlapping late-training windows, after update-support embargo',validation_rule='Evenly spaced nonoverlapping val windows'))
    write(out/'update_schedule.json',schedule);write(out/'settings.json',opts)
    before,_=evaluate(model,ds,updates,device,opts['batch_size'],mean,opts['fusion_weights'],out/'update_before')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    objective=ReconstructionObjective(opts['loss_weights']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=opts['learning_rate']);history=[]
    for step,ids in enumerate(schedule,1):
        model.train();optimizer.zero_grad(set_to_none=True)
        batch={k:v[ids].to(device) for k,v in data.items()};result=model(batch['X'])
        loss,levels=objective(batch['values'],batch['status'],result['reconstructions'])
        require(torch.isfinite(loss).item(),'Nonfinite reconstruction loss');loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),opts['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),levels={k:float(v.detach()) for k,v in levels.items()}));report['additional_steps']=step
        if step==1 or step%64==0:print(f'Reconstruction {step}/{opts["steps"]}: loss={float(loss.detach()):.6g}',flush=True)
    write(out/'history.json',history)
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'Nonfinite final model')
    train_summary,_=evaluate(model,ds,updates,device,opts['batch_size'],mean,opts['fusion_weights'],out/'update_after')
    cal_summary,cal_score= evaluate(model,ds,calibration,device,opts['batch_size'],mean,opts['fusion_weights'],out/'calibration_scores')
    threshold=fit_threshold(cal_score,opts['threshold_quantile'])
    # Fixed checkpoint and threshold; validation cannot select an epoch or fit a threshold.
    cp=dict(protocol='bre_input_reconstruction_v1',objective='masked_input_window_mse',model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        configuration=config,semantics=semantics,settings=opts,channels=ds.devices,mean_values=mean.cpu(),source_hashes=dependencies,
        input_metadata_hashes=ds.metadata_hashes,input_manifest_sha256=ds.manifest_sha256,parent_encoder_sha256=sha(root/'encoder_checkpoint.pt'),
        previous_reconstruction_steps=512,additional_steps=opts['steps'],selection_sha256=sha(out/'selection.json'))
    torch.save(cp,out/'reconstruction_checkpoint.pt')
    write(out/'calibration.json',dict(protocol='bre_reconstruction_calibration_v1',checkpoint_sha256=sha(out/'reconstruction_checkpoint.pt'),
        quantile=opts['threshold_quantile'],threshold=threshold,comparison='strict_greater',calibration_split='train',calibration_windows=len(calibration),
        calibration_selection_sha256=sha(out/'selection.json'),fusion_weights=opts['fusion_weights'],interpretation='Development threshold; no guaranteed false-positive rate'))
    val_summary,val_score=evaluate(model,val,validation,device,opts['batch_size'],mean,opts['fusion_weights'],out/'validation_scores')
    ds.verify_unchanged();val.verify_unchanged();check_sources(dependencies)
    deltas={name:float(sum((v.detach().cpu().double()-initial[k].double()).square().sum() for k,v in model.named_parameters() if k.startswith(name+'.')).sqrt()) for name in ('dbrl','bse','bil','ebrl','reconstructor')}
    report.update(status='COMPLETE',settings=opts,device=str(device),parent_report_sha256=sha(args.training_report),migration_replay_max_error=migration_error,representation_replay_max_error=representation_replay_error,same_device_encoder_migration_exact=True,
        input_manifest_sha256=ds.manifest_sha256,parameter_delta_l2=deltas,update_before=before,update_after=train_summary,
        calibration=cal_summary,validation=val_summary,threshold=threshold,calibration_exceedance_fraction=float(np.mean(cal_score>threshold)),
        validation_exceedance_fraction=float(np.mean(val_score>threshold)),checkpoint_sha256=sha(out/'reconstruction_checkpoint.pt'),
        source_hashes=dependencies,runner_sha256=sha(__file__),torch_version=str(torch.__version__),
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir');p.add_argument('--input_dir',type=Path,required=True)
    p.add_argument('--training_report',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda']);args=p.parse_args()
    config=load_config(args.project_root_dir);out=new_output(config,'bre_reconstruction')
    report=dict(status='RUNNING',protocol='bre_reconstruction_training_v1',additional_steps=0,scientific_status='DEVELOPMENT_ONLY',
        limitations=['Input reconstruction is not established behavior discrimination or anomaly detection.','Validation exceedance is not a labeled false-positive rate.',
            'Calibration is train-period development data; scaler used full training period.','Master-file imputation and single-neighbor BIL limitations remain.',
            'Input-space discrepancies replace latent forecasting discrepancies; manuscript methodology must be updated.'])
    write(out/'configuration_snapshot.json',config)
    try:run(args,config,out,report)
    except Exception as e:report.update(status='BLOCKED',error=type(e).__name__+': '+str(e))
    write(out/'reconstruction_report.json',report);print('BRE reconstruction:',report['status']);print('Report:',out/'reconstruction_report.json')
    if report['status']!='COMPLETE':print(report['error']);raise SystemExit(2)

if __name__=='__main__':main()
