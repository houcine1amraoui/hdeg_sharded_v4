"""Bounded reconstruction scoring with a fixed checkpoint and train-calibrated threshold."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.data_preparation.bre_windows import BREWindowDataset
from src.models.hdeg.bre_reconstruction import BREReconstructionModel
from src.scientific_validation.reconstruction.data import evenly_spaced_starts
from src.scientific_validation.reconstruction.runtime import (
    require,sha,write,load_config,configure_device,new_output,check_sources,evaluate)


def run(args,config,out,report):
    # Require the successful training report to bind both checkpoint and calibration bytes.
    parent=json.loads(args.reconstruction_report.read_text());root=args.reconstruction_report.parent
    require(parent['status']=='COMPLETE' and parent['protocol']=='bre_reconstruction_training_v1','Expected completed reconstruction report')
    for name in ('reconstruction_checkpoint.pt','calibration.json','selection.json'):
        require(sha(root/name)==parent['output_hashes'][name],'Reconstruction artifact differs: '+name)
    cp=torch.load(root/'reconstruction_checkpoint.pt',map_location='cpu',weights_only=True)
    require(cp['protocol']=='bre_input_reconstruction_v1','Forecasting checkpoints are not reconstruction checkpoints')
    cal=json.loads((root/'calibration.json').read_text())
    require(cal['protocol']=='bre_reconstruction_calibration_v1' and cal['checkpoint_sha256']==sha(root/'reconstruction_checkpoint.pt'),'Calibration/checkpoint mismatch')
    require(cal['fusion_weights']==cp['settings']['fusion_weights'] and cal['comparison']=='strict_greater','Calibration scoring rule differs')
    require(cal['calibration_selection_sha256']==cp['selection_sha256']==sha(root/'selection.json'),'Calibration selection differs')
    for key in ('hdeg','preprocessing','seed'):
        require(config.get(key)==cp['configuration'].get(key),'Configuration differs: '+key)
    ds=BREWindowDataset(args.input_dir,args.split,config['preprocessing']['window_size'])
    require(ds.metadata_hashes==cp['input_metadata_hashes'],'Input metadata differs')
    require(ds.devices==cp['channels'],'Channel order differs');check_sources(cp['source_hashes'])
    require(type(args.windows)is int and 1<=args.windows<=8192,'Bounded inference supports 1..8192 windows')
    starts=evenly_spaced_starts(0,ds.num_rows,ds.window_size,args.windows)
    write(out/'selection.json',dict(split=args.split,indices=starts,window_size=ds.window_size))
    device=configure_device(config,args.device);model=BREReconstructionModel(config,cp['semantics']).to(device);model.load_state_dict(cp['model'],strict=True)
    summary,scores=evaluate(model,ds,starts,device,cp['settings']['batch_size'],cp['mean_values'].to(device),cp['settings']['fusion_weights'],out/'scores')
    threshold=cal['threshold'];flags=scores>threshold
    np.savez_compressed(out/'decisions.npz',indices=np.array(starts),score=scores,exceeds_threshold=flags,threshold=np.array(threshold))
    ds.verify_unchanged();check_sources(cp['source_hashes'])
    report.update(status='COMPLETE',split=args.split,summary=summary,threshold=threshold,exceedance_fraction=float(flags.mean()),
        checkpoint_sha256=sha(root/'reconstruction_checkpoint.pt'),calibration_sha256=sha(root/'calibration.json'),
        runner_sha256=sha(__file__),output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir');p.add_argument('--input_dir',type=Path,required=True)
    p.add_argument('--reconstruction_report',type=Path,required=True);p.add_argument('--split',choices=['val','actor1_test','actor2_test'],default='val')
    p.add_argument('--windows',type=int,default=1024);p.add_argument('--device',choices=['cpu','cuda']);args=p.parse_args()
    config=load_config(args.project_root_dir);out=new_output(config,'bre_reconstruction_scores')
    report=dict(status='RUNNING',protocol='bre_reconstruction_scoring_v1',interpretation='Window scores and threshold exceedances, not labeled anomaly metrics')
    try:run(args,config,out,report)
    except Exception as e:report.update(status='BLOCKED',error=type(e).__name__+': '+str(e))
    write(out/'scoring_report.json',report);print('Reconstruction scoring:',report['status']);print('Report:',out/'scoring_report.json')
    if report['status']!='COMPLETE':print(report['error']);raise SystemExit(2)

if __name__=='__main__':main()
