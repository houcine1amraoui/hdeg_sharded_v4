"""Shared reconstruction execution: configuration, lineage, evaluation, checkpoints."""
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import uuid
import numpy as np
import torch
import yaml
from src.models.hdeg.bre_reconstruction import BREReconstructionModel
from src.models.hdeg.reconstruction import LEVELS, ReconstructionAnomalyInference
from src.scientific_validation.reconstruction.data import window_batch


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False)


def load_config(project_root):
    with open('configs/config.yaml') as f: config = yaml.safe_load(f)
    if project_root: config['project_root_dir'] = project_root
    config = copy.deepcopy(config)
    config['preprocessing']['dataset_name'] = 'BRE'; config['preprocessing']['merge_bre_cu'] = False
    return config


def settings(config):
    s = dict(hidden_dim=64, steps=512, batch_size=8, learning_rate=1e-4, gradient_clip=1.,
             calibration_windows=1024, validation_windows=1024, threshold_quantile=.99,
             loss_weights=None, fusion_weights=None)
    override = config['hdeg'].get('reconstruction', {})
    require(not set(override) - set(s), 'Unknown hdeg.reconstruction settings')
    s.update(override)
    for key in ('hidden_dim','steps','batch_size','calibration_windows','validation_windows'):
        require(type(s[key]) is int and s[key] > 0, 'Invalid setting: ' + key)
    require(s['hidden_dim'] == 64, 'Migrated decoders require hidden_dim=64')
    require(s['steps'] <= 4096 and s['batch_size'] <= 256 and max(s['calibration_windows'],s['validation_windows']) <= 8192,
            'Bounded runner budget exceeded')
    require(s['learning_rate'] > 0 and np.isfinite(s['learning_rate']) and s['gradient_clip'] > 0 and np.isfinite(s['gradient_clip']), 'Invalid optimization settings')
    require(0 < s['threshold_quantile'] < 1, 'Invalid threshold quantile')
    return s


def configure_device(config, requested=None):
    device = torch.device(requested or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(config.get('seed', 42)); np.random.seed(config.get('seed', 42)); torch.set_num_threads(1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.get('seed', 42))
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    return device


def new_output(config, name):
    from src.utils.get_folders_utils import get_processed_folder
    out = Path(get_processed_folder(config)) / 'v2_validation' / name / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    return out


def source_hashes(model):
    # Hash every HDEG module used, including inherited encoder classes and graph builder.
    paths = {Path(__file__), Path(inspect.getfile(BREReconstructionModel)),
        Path(inspect.getfile(ReconstructionAnomalyInference))}
    from src.scientific_validation.reconstruction import data
    from src.data_preparation import bre_windows
    from src.models.hdeg import bre_semantics, dbrl
    from src.common.graph import graph_builder
    for module in (data, bre_windows, bre_semantics, dbrl, graph_builder): paths.add(Path(inspect.getfile(module)))
    for module in model.modules():
        if module.__class__.__module__.startswith('src.'):
            paths.add(Path(inspect.getfile(type(module))))
    paths.add(Path('configs/hdeg/bre_semantics.yaml'))
    result = {}
    for p in paths:
        s = str(p.resolve()); key = 'src/' + s.split('/src/',1)[1] if '/src/' in s else 'configs/hdeg/bre_semantics.yaml'
        result[key] = sha(p)
    return result


def check_sources(expected):
    for name, digest in expected.items(): require(sha(name) == digest, 'Source differs: ' + name)


def evaluate(model, ds, starts, device, batch_size, mean_values, fusion_weights, destination):
    """Bounded evaluation. Reports scores, counts, channel/group errors and variation."""
    model.eval(); scorer = ReconstructionAnomalyInference(fusion_weights).to(device)
    channel_rows = []; channel_count_rows = []; level_errors = []; scores = []; baseline_errors = []; source = []; times = []
    channel_sums = np.zeros((4, model.num_devices)); channel_counts = np.zeros(model.num_devices)
    rep_sums = {}; rep_square = {}; count = 0
    with torch.no_grad():
        for start in range(0, len(starts), batch_size):
            ids = starts[start:start + batch_size]; b = window_batch(ds, ids)
            x = torch.from_numpy(b['X']).to(device); values = torch.from_numpy(b['values']).to(device); status = torch.from_numpy(b['status']).to(device)
            out = model(x); evidence = scorer(values, status, out['reconstructions'])
            err = torch.stack([evidence['levels'][k] for k in LEVELS], 1).cpu().numpy()
            valid = status == 0; counts = valid.sum((1,2))
            baseline = (torch.where(valid,(values-mean_values).square(),torch.zeros_like(values)).sum((1,2))/counts).cpu().numpy()
            cc = evidence['channel_counts'].cpu().numpy()
            channel_rows.append(np.stack([evidence['channels'][k].cpu().numpy() for k in LEVELS],1));channel_count_rows.append(cc)
            channel_counts += cc.sum(0)
            for j,k in enumerate(LEVELS): channel_sums[j] += (evidence['channels'][k].cpu().numpy()*cc).sum(0)
            for k,v in out['representations'].items():
                a = v.double().flatten(1).cpu().numpy()
                rep_sums[k] = rep_sums.get(k,0) + a.sum(0)
                rep_square[k] = rep_square.get(k,0) + (a*a).sum(0)
            count += len(ids); level_errors.append(err); baseline_errors.append(baseline)
            scores.append(evidence['score'].cpu().numpy()); source.extend(zip(b['source_start'].tolist(),b['source_stop'].tolist()));times.extend(b['timestamp_end'].astype(str).tolist())
    errors = np.concatenate(level_errors); score = np.concatenate(scores); baseline = np.concatenate(baseline_errors)
    per_channel = np.divide(channel_sums,channel_counts[None],out=np.zeros_like(channel_sums),where=channel_counts[None]>0)
    variation = {k:float(np.maximum(rep_square[k]/count-(rep_sums[k]/count)**2,0).mean()) for k in LEVELS}
    summary = dict(windows=count, mean_baseline_mse=float(baseline.mean()), levels={}, representation_variance=variation,
        channels=ds.devices, channel_valid_counts=channel_counts.tolist(), channel_mse={k:[float(v) if channel_counts[j]>0 else None for j,v in enumerate(per_channel[i])] for i,k in enumerate(LEVELS)})
    for i,k in enumerate(LEVELS):
        summary['levels'][k] = dict(mse=float(errors[:,i].mean()), improvement_vs_mean=float(1-errors[:,i].mean()/baseline.mean()) if baseline.mean()>0 else None,
                                    win_count=int((errors[:,i]<baseline).sum()))
    summary['semantic_group_mse'] = {}
    for j, state in enumerate(model.state_ids):
        member = model.compatibility_mask[j].cpu().numpy().astype(bool)
        denom = channel_counts[member].sum()
        summary['semantic_group_mse'][state] = {k: float(channel_sums[i,member].sum()/denom) if denom>0 else None for i,k in enumerate(LEVELS)}
    np.savez_compressed(destination.with_suffix('.npz'),channel_mse=np.concatenate(channel_rows),channel_valid_counts=np.concatenate(channel_count_rows), indices=np.array(starts), levels=errors, score=score, mean_baseline=baseline,
                        source_ranges=np.array(source), timestamp_end=np.array(times))
    write(destination.with_suffix('.json'), summary)
    return summary, score
