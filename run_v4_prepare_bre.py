"""Chunked preparation of the dataset-provided BREMaster.csv; no model training."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
import numpy as np
import pandas as pd
import yaml

POLICY = 'bre_master_input_v1'
SPLITS = ('train', 'val', 'actor2_test', 'actor1_test')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(path):
    s = Path(path).stat()
    return (s.st_size, s.st_mtime_ns)


def registry(selected):
    result = []
    for index, name in enumerate(selected):
        kind = re.sub(r'_[xy]$', '', name).rsplit('_', 1)[-1]
        require(kind in ('contact','motion','temperature','humidity','illuminance'), 'Unsupported selected feature: '+name)
        result.append(dict(index=index, id=name, measurement=kind,
            value_type='binary' if kind in ('contact','motion') else 'continuous'))
    return result


def parse_values(frame, channels):
    """Status reflects CSV content, not original sensor availability before imputation."""
    values = np.zeros((len(frame),len(channels)), np.float64)
    status = np.full(values.shape, 4, np.uint8)
    for j, ch in enumerate(channels):
        text = frame[ch['id']].str.strip().str.lower()
        missing = text.isin(['','nan','null','none','na','n/a']).to_numpy()
        unknown = text.eq('unknown').to_numpy()
        unavailable = text.eq('unavailable').to_numpy()
        numeric = pd.to_numeric(text, errors='coerce').to_numpy(dtype=np.float64)
        valid = np.isfinite(numeric) & ~(missing|unknown|unavailable)
        if ch['value_type']=='binary':valid &= (numeric==0)|(numeric==1)
        values[valid,j] = numeric[valid]
        status[missing,j]=3;status[unknown,j]=1;status[unavailable,j]=2;status[valid,j]=0
    return values,status


def periods(config):
    a=config['preprocessing']['actors_timelines'];result=[]
    for key in ('actor1_t1','actor2','actor1_t2'):
        start=pd.Timestamp(a[key+'_start']);end=pd.Timestamp(a[key+'_end'])
        require(start.tz is None and end.tz is None and start<=end, 'Invalid timezone-naive period: '+key)
        result.append((start.value,end.value))
    require(result[0][1]<result[1][0] and result[1][1]<result[2][0], 'Actor periods overlap or are not chronological')
    return result


def timestamps(frame):
    ts=pd.to_datetime(frame['Timestamp'],format='%Y-%m-%d %H:%M:%S',errors='raise')
    return ts.to_numpy(dtype='datetime64[ns]').astype(np.int64)


def chunks(path, selected, size):
    with pd.read_csv(path,usecols=['Timestamp']+selected,dtype=str,keep_default_na=False,
                     encoding='utf-8-sig',chunksize=size) as reader:
        yield from reader


def prepare(path, selected_file, config, out, chunk_rows=32768, shard_rows=8192):
    require(1<=chunk_rows<=262144 and 1<=shard_rows<=65536, 'Chunk/shard size exceeds bound')
    require(config['preprocessing'].get('downsample_freq')=='1s', 'This runner preserves the supplied 1-second grid; use downsample_freq: 1s')
    before=signature(path);source_hash=sha(path);feature_hash=sha(selected_file)
    selected=[line.strip() for line in Path(selected_file).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(len(selected)==62 and len(set(selected))==62, 'Expected 62 distinct selected BRE features')
    with Path(path).open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f))
    require(len(header)==len(set(header)), 'Duplicate CSV headers')
    require('Timestamp' in header and all(x in header for x in selected), 'Timestamp or selected features missing; no silent filtering')
    channels=registry(selected);ranges=periods(config);ratio=config['preprocessing']['val_ratio']
    require(isinstance(ratio,(float,int)) and 0<ratio<1, 'val_ratio must be between zero and one')
    counts=np.zeros(3,np.int64);rows=0;previous=None;first=None
    print('Pass 1/3: validating timestamps and determining chronological split sizes',flush=True)
    # Only timestamp values are inspected in pass 1; no holdout statistics inform fitting.
    for chunk in chunks(path,[],chunk_rows):
        t=timestamps(chunk)
        require(len(t)>0 and np.all(t%1_000_000_000==0), 'Subsecond timestamp in one-second master')
        diffs=np.diff(t) if previous is None else np.diff(np.r_[previous,t])
        require(np.all(diffs==1_000_000_000), 'Timestamps contain duplicates, gaps or disorder; no silent sorting/filling')
        if first is None:first=int(t[0])
        previous=int(t[-1]);rows+=len(t)
        for j,(lo,hi) in enumerate(ranges):counts[j]+=int(((t>=lo)&(t<=hi)).sum())
    require(np.all(counts>0),'One or more configured actor periods has no data')
    ntrain=int(int(counts[0])*(1-ratio));require(0<ntrain<counts[0], 'Empty train or validation split')
    expected=dict(zip(SPLITS,[ntrain,int(counts[0])-ntrain,int(counts[1]),int(counts[2])]))
    require(signature(path)==before,'Source changed during pass 1')
    def split_masks(t, seen):
        masks=[(t>=lo)&(t<=hi) for lo,hi in ranges]
        ranks=seen+np.cumsum(masks[0])-1
        return dict(train=masks[0]&(ranks<ntrain),val=masks[0]&(ranks>=ntrain),actor2_test=masks[1],actor1_test=masks[2]),seen+int(masks[0].sum())
    low=np.full(62,np.inf);high=np.full(62,-np.inf);valid_counts=np.zeros(62,np.int64);seen=0
    train_status=np.zeros((62,5),np.int64)
    print('Pass 2/3: fitting continuous scaling on training rows only',flush=True)
    for frame in chunks(path,selected,chunk_rows):
        masks,seen=split_masks(timestamps(frame),seen);train=frame.loc[masks['train']]
        if not len(train):continue
        v,s=parse_values(train,channels)
        for j in range(62):
            a=v[s[:,j]==0,j];valid_counts[j]+=len(a)
            train_status[j]+=np.bincount(s[:,j],minlength=5)
            if len(a):low[j]=min(low[j],float(a.min()));high[j]=max(high[j],float(a.max()))
    require(np.all(valid_counts>0),'Selected channels have no valid training values: '+str([selected[j] for j in np.flatnonzero(valid_counts==0)]))
    constant=high==low;den=np.where(constant,1.,high-low)
    binary=np.array([c['value_type']=='binary' for c in channels])
    offset=np.where(binary,0.,low);den=np.where(binary,1.,den)
    require(signature(path)==before,'Source changed during pass 2')
    scaler=dict(fit_split='train',fit_rows=ntrain,channels=selected,min=low.tolist(),max=high.tolist(),
        offset=offset.tolist(),denominator=den.tolist(),valid_counts=valid_counts.tolist(),
        constant_in_train=constant.tolist(),rule='Binary unchanged; continuous (value-train_min)/range; constant range uses 1. No clipping.')
    write(out/'scaler.json',scaler);write(out/'devices.json',selected)
    write(out/'input_state_registry.json',dict(policy_version=POLICY,channels=channels,
        status_codes={'0':'numeric_valid_in_master','1':'unknown_token','2':'unavailable_token','3':'missing_token','4':'invalid_value'},
        features=['value','valid','unknown','unavailable','missing','invalid'],
        limitation='Status describes supplied CSV cells. Previously filled values cannot be distinguished from original observations.'))
    write(out/'configuration_snapshot.json',config)
    manifests={k:dict(split=k,num_rows=0,shards=[],status_counts=np.zeros((62,5),np.int64),outside_train_range=np.zeros(62,np.int64)) for k in SPLITS}
    for k in SPLITS:(out/k).mkdir()
    print('Pass 3/3: transforming and writing disjoint split shards',flush=True)
    seen=0;source_row=0
    for chunk_number,frame in enumerate(chunks(path,selected,chunk_rows),1):
        t=timestamps(frame);masks,seen=split_masks(t,seen)
        for split,mask in masks.items():
            positions=np.flatnonzero(mask)
            if not len(positions):continue
            v,s=parse_values(frame.iloc[positions],channels);good=s==0
            scaled=np.where(good,(v-offset)/den,0.)
            require(np.isfinite(scaled).all() and np.max(np.abs(scaled))<=np.finfo(np.float32).max, 'Scaled values overflow float32')
            m=manifests[split]
            m['outside_train_range']+=(((v<low)|(v>high))&good).sum(0)
            for j in range(62):m['status_counts'][j]+=np.bincount(s[:,j],minlength=5)
            for start in range(0,len(positions),shard_rows):
                sl=slice(start,start+shard_rows);pos=positions[sl];name=f'shard_{len(m["shards"]):06d}.npz';dest=out/split/name
                np.savez_compressed(dest,values=scaled[sl].astype(np.float32),status=s[sl],
                    timestamps=t[pos].astype('datetime64[ns]'),source_rows=(source_row+pos).astype(np.int64))
                m['shards'].append(dict(file=name,start=m['num_rows'],stop=m['num_rows']+len(pos),
                    source_row_start=int(source_row+pos[0]),source_row_stop=int(source_row+pos[-1]+1),
                    first_timestamp=str(t[pos[0]].astype('datetime64[ns]')),last_timestamp=str(t[pos[-1]].astype('datetime64[ns]')),sha256=sha(dest)))
                m['num_rows']+=len(pos)
        source_row+=len(frame)
        if chunk_number%10==0:print('Rows processed:',source_row,flush=True)
    require(signature(path)==before and sha(path)==source_hash,'Source changed during preparation')
    require(sha(selected_file)==feature_hash,'Selected feature file changed during preparation')
    window=config['preprocessing'].get('window_size',30);require(type(window) is int and window>0,'Invalid window size')
    for split,m in manifests.items():
        require(m['num_rows']==expected[split], 'Split count changed: '+split)
        m['status_counts']=m['status_counts'].tolist();m['outside_train_range']=m['outside_train_range'].tolist()
        m.update(policy_version=POLICY,devices=selected,window_size=window,
            possible_one_step_pairs=max(0,m['num_rows']-window),
            window_rule='Construct X and shifted Y wholly within this split; cross-shard reads allowed, cross-split reads forbidden.')
        write(out/split/'manifest.json',m)
    report=dict(status='COMPLETE',policy_version=POLICY,scientific_status='EXPLORATORY_MASTER_BASED_PREPARATION',
        source=dict(path=str(Path(path).resolve()),bytes=before[0],mtime_ns=before[1],sha256=source_hash,
            provenance='User confirms dataset-provided BREMaster.csv; only feature selection subsequently applied.'),
        runner_sha256=sha(__file__),execution=dict(chunk_rows=chunk_rows,shard_rows=shard_rows,numpy=np.__version__,pandas=pd.__version__),
        selected_features_sha256=feature_hash,selected_count=62,source_rows=rows,excluded_rows=rows-int(counts.sum()),
        split_counts=expected,train_status_counts=train_status.tolist(),constant_training_channels=[selected[j] for j in np.flatnonzero(constant)],
        first_timestamp=str(np.datetime64(first,'ns')),last_timestamp=str(np.datetime64(previous,'ns')),
        limitations=['Original acquisition/imputation status is not recoverable from master cells.',
            'Actor date ranges come from the existing configuration; this runner does not independently verify activity labels or clock alignment.',
            'No resampling, forward/backward filling, clipping, feature dropping, training or window materialization.',
            'BRE metadata is separate from CU; no CU semantic mapping is reused.'],
        artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')})
    write(out/'preparation_report.json',report)
    write(out/'manifest.json',dict(status='COMPLETE',policy_version=POLICY,devices=selected,
        splits={k:k+'/manifest.json' for k in SPLITS},scaler='scaler.json',registry='input_state_registry.json',
        source_sha256=source_hash,report_sha256=sha(out/'preparation_report.json')))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir');p.add_argument('--chunk_rows',type=int);p.add_argument('--shard_rows',type=int)
    args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    config=copy.deepcopy(config);config['preprocessing']['dataset_name']='BRE';config['preprocessing']['merge_bre_cu']=False
    from src.utils.get_folders_utils import get_dataset_path,get_processed_folder
    root=Path(config['project_root_dir']);base=root/config['preprocessing']['dataset_folder']/'processed'/'BRE';base.mkdir(parents=True,exist_ok=True)
    out=Path(get_processed_folder(config))/'input_states'/POLICY/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    settings=config.get('bre_preparation',{})
    try:
        result=prepare(Path(get_dataset_path(config)),Path('configs/BRE_selected_features.txt'),config,out,
            args.chunk_rows if args.chunk_rows is not None else settings.get('chunk_rows',32768),args.shard_rows if args.shard_rows is not None else settings.get('shard_rows',8192))
        print('BRE preparation:',result['status']);print('Split rows:',result['split_counts'])
    except Exception as e:
        write(out/'failure_report.json',dict(status='BLOCKED',error=type(e).__name__+': '+str(e),partial_outputs_not_usable=True))
        print('BRE preparation: BLOCKED');print(type(e).__name__+': '+str(e));print('Report:',out/'failure_report.json');raise SystemExit(2)
    print('Report:',out/'preparation_report.json');print('Input directory:',out)

if __name__=='__main__':main()
