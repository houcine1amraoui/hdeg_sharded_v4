"""Bounded cached reads of bre_master_input_v1 artifacts, confined to one split."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


class BREWindowDataset:
    def __init__(self, input_dir, split='train', window_size=30, cache_shards=2):
        self.root=Path(input_dir).resolve();self.split=split;self.cache=OrderedDict();self.loaded_hashes={}
        require(type(window_size) is int and window_size>0,'Invalid window size')
        require(type(cache_shards) is int and cache_shards>0,'Invalid cache bound')
        self.window_size=window_size;self.cache_shards=cache_shards;self.metadata_hashes={}
        def read(name, expected=None):
            p=self.root/name;digest=sha(p)
            require(expected is None or digest==expected,'Metadata hash mismatch: '+name)
            self.metadata_hashes[name]=digest
            return json.loads(p.read_text())
        root=read('manifest.json');require(root['status']=='COMPLETE' and root['policy_version']=='bre_master_input_v1','Wrong preparation contract')
        report=read('preparation_report.json',root['report_sha256'])
        require(report['status']=='COMPLETE' and report['source']['sha256']==root['source_sha256'],'Preparation lineage mismatch')
        # Fixed names prevent path redirection; all split metadata is checked, not test values.
        require(root['splits']=={k:k+'/manifest.json' for k in ('train','val','actor2_test','actor1_test')},'Unexpected split paths')
        require(split in root['splits'],'Unknown split')
        self.devices=read('devices.json',report['artifacts']['devices.json'])
        self.registry=read('input_state_registry.json',report['artifacts']['input_state_registry.json'])
        self.scaler=read('scaler.json',report['artifacts']['scaler.json'])
        self.configuration=read('configuration_snapshot.json',report['artifacts']['configuration_snapshot.json'])
        require(self.devices==root['devices']==self.scaler['channels']==[c['id'] for c in self.registry['channels']],'Channel order mismatch')
        require(len(self.devices)>0 and len(set(self.devices))==len(self.devices),'Duplicate/empty device order')
        require(self.registry['features']==['value','valid','unknown','unavailable','missing','invalid'],'Input feature order mismatch')
        previous_source=None
        for key,name in root['splits'].items():
            m=read(name,report['artifacts'][name]);end=0
            require(m['devices']==self.devices and m['policy_version']=='bre_master_input_v1','Split metadata mismatch')
            for e in m['shards']:
                require(Path(e['file']).name==e['file'] and e['file'].endswith('.npz'),'Unsafe shard path')
                require(e['start']==end and e['stop']>end,'Noncontiguous shard ranges')
                require(e['source_row_stop']-e['source_row_start']==e['stop']-e['start'],'Source range mismatch')
                require(previous_source is None or e['source_row_start']>=previous_source,'Overlapping source ranges')
                previous_source=e['source_row_stop'];end=e['stop']
            require(end==m['num_rows']==report['split_counts'][key],'Split row count mismatch')
            if key==split:self.manifest=m
        self.num_rows=self.manifest['num_rows'];self.shards=self.manifest['shards']
        require(self.window_size==self.manifest['window_size'],'Window size differs from preparation')
        self.stops=np.array([s['stop'] for s in self.shards]);self.manifest_sha256=self.metadata_hashes['manifest.json']
        self.binary=np.array([c['value_type']=='binary' for c in self.registry['channels']])

    def __len__(self): return max(0,self.num_rows-self.window_size)

    def load_shard(self, index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        e=self.shards[index];path=self.root/self.split/e['file'];raw=path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==e['sha256'],'Shard hash mismatch: '+str(path))
        with np.load(io.BytesIO(raw),allow_pickle=False) as f: a={k:f[k] for k in ('values','status','timestamps','source_rows')}
        n=e['stop']-e['start'];v=a['values'];s=a['status'];t=a['timestamps'];sr=a['source_rows']
        require(v.shape==s.shape==(n,len(self.devices)),'Shard dimensions differ')
        require(v.dtype==np.float32 and s.dtype==np.uint8 and t.dtype==np.dtype('datetime64[ns]') and sr.dtype==np.int64,'Shard dtype mismatch')
        require(t.shape==sr.shape==(n,) and np.isfinite(v).all() and np.all(s<=4),'Invalid shard values')
        require(np.all(v[s!=0]==0),'Nonvalid placeholders must be zero')
        require(np.all((s[:,self.binary]!=0)|(v[:,self.binary]==0)|(v[:,self.binary]==1)),'Invalid binary values')
        require(np.array_equal(sr,np.arange(e['source_row_start'],e['source_row_stop'])),'Source row alignment differs')
        require(str(t[0])==e['first_timestamp'] and str(t[-1])==e['last_timestamp'] and np.all(np.diff(t.astype(np.int64))==10**9),'Timestamp alignment differs')
        for x in a.values():x.flags.writeable=False
        self.cache[index]=a;self.loaded_hashes[str(path)]=e['sha256']
        while len(self.cache)>self.cache_shards:self.cache.popitem(last=False)
        return a

    def read_rows(self,start,stop):
        require(type(start) in (int,np.int64) and type(stop) in (int,np.int64) and 0<=start<stop<=self.num_rows,'Read crosses split boundary or has invalid indices')
        parts=[];cursor=int(start)
        while cursor<stop:
            j=int(np.searchsorted(self.stops,cursor,side='right'));e=self.shards[j];end=min(stop,e['stop']);a=self.load_shard(j)
            parts.append({k:v[cursor-e['start']:end-e['start']] for k,v in a.items()});cursor=end
        result={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        require(np.all(np.diff(result['source_rows'])==1) and np.all(np.diff(result['timestamps'].astype(np.int64))==10**9),'Discontinuous cross-shard read')
        return result

    def get_batch(self,indices):
        indices=list(indices);require(0<len(indices)<=256,'Batch must contain 1..256 windows')
        require(all(isinstance(i,(int,np.integer)) and not isinstance(i,(bool,np.bool_)) and 0<=i<len(self) for i in indices),'Invalid pair index; would cross split boundary')
        blocks=[self.read_rows(int(i),int(i)+self.window_size+1) for i in indices]
        b={k:np.stack([a[k] for a in blocks]) for k in blocks[0]}
        features=np.concatenate([b['values'][...,None],np.eye(5,dtype=np.float32)[b['status']]],axis=-1)
        result={'indices':np.array(indices,np.int64),'X':features[:,:-1].copy(),'Y':features[:,1:].copy()}
        for side,sl in [('X',slice(None,-1)),('Y',slice(1,None))]:
            for key in b: result[side+'_'+key]=b[key][:,sl].copy()
        return result

    def verify_unchanged(self):
        for name,h in self.metadata_hashes.items(): require(sha(self.root/name)==h,'Metadata changed: '+name)
        for name,h in self.loaded_hashes.items(): require(sha(name)==h,'Consumed shard changed: '+name)
