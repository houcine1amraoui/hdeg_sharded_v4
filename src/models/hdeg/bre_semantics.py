"""BRE semantic grouping derived from registry measurement types, preserving exact IDs."""
import numpy as np
import yaml


def load_bre_semantics(path, registry):
    with open(path) as f:spec=yaml.safe_load(f)
    if spec.get('schema_version')!='bre_semantics_v1' or spec.get('dataset')!='BRE':raise ValueError('Wrong semantic specification')
    states=spec['states'];ids=[s['id'] for s in states];kinds=[s['measurement'] for s in states]
    if not ids or len(set(ids))!=len(ids) or len(set(kinds))!=len(kinds):raise ValueError('Empty or duplicate states')
    channels=registry['channels'];mask=np.array([[c['measurement']==kind for c in channels] for kind in kinds],dtype=bool)
    if not mask.any(1).all() or not np.all(mask.sum(0)==1):raise ValueError('Every state needs members and every channel exactly one state')
    edges=[(ids.index(a),ids.index(b)) for a,b in spec['edges']]
    if len(set(edges))!=len(edges) or any(a==b for a,b in edges):raise ValueError('Duplicate or self edge')
    result=dict(spec)
    result.update(state_ids=ids,device_ids=[c['id'] for c in channels],compatibility_mask=mask.tolist(),
        edge_index=np.array(edges,dtype=np.int64).reshape(-1,2).T.tolist(),
        member_counts=mask.sum(1).tolist())
    return result
