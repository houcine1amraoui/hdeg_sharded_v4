"""Controlled 1/10/30-second BRE horizon comparison; train-period development only."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import numpy as np
import yaml
from run_v4_train_bre import require, sha, write

LEVELS = ('Z', 'S', 'S_tilde', 'g')
HORIZONS = (1, 10, 30)
FAMILIES = ('quiet', 'older_change', 'future_change')


def classify(changes, starts, horizon, window=30):
    """Edge j links source rows j and j+1. Includes all newly arriving edges."""
    c = np.r_[0, np.cumsum(changes, dtype=np.int64)]
    starts = np.asarray(starts, dtype=np.int64)
    old = c[starts + window - 1] - c[starts]
    future = c[starts + window + horizon - 1] - c[starts + window - 1]
    return np.where(future > 0, 'future_change', np.where(old > 0, 'older_change', 'quiet'))


def choose(changes, lower, upper, quotas, seed, excluded=()):
    """Disjoint 60-row supports, independent of horizon-specific target values."""
    starts = np.arange(lower, upper - 59, dtype=np.int64)
    labels = classify(changes, starts, 30)
    blocked = np.zeros(len(changes) + 1, dtype=bool)
    # excluded contains inclusive intervals, e.g. previously encoder-fitted X supports.
    for lo, hi in excluded:
        blocked[max(0, lo - 59):min(len(blocked), hi + 1)] = True
    rng = np.random.default_rng(seed)
    selected, counts = [], {}
    for family in ('future_change', 'older_change', 'quiet'):
        candidates = starts[labels == family]
        counts[family] = int(len(candidates))
        chosen = []
        for i in rng.permutation(candidates):
            i = int(i)
            if blocked[i]:
                continue
            chosen.append(i)
            blocked[max(0, i - 59):i + 60] = True
            if len(chosen) == quotas[family]:
                break
        require(len(chosen) == quotas[family], 'Insufficient disjoint ' + family + ' supports')
        selected.extend(dict(index=i, family=family) for i in sorted(chosen))
    return selected, counts


def pair_inputs(ds, starts):
    """Read exact union support once, including cross-shard continuity checks."""
    blocks = [ds.read_rows(int(i), int(i) + 60) for i in starts]
    values = np.stack([b['values'] for b in blocks])
    status = np.stack([b['status'] for b in blocks])
    require(np.all(status == 0), 'Unexpected nonvalid master cells')
    features = np.concatenate([values[..., None], np.eye(5, dtype=np.float32)[status]], -1)
    return features[:, :30].copy(), {h: features[:, h:h + 30].copy() for h in HORIZONS}


def summarize(records):
    summary = {}
    for family in ('all',) + FAMILIES:
        group = [r for r in records if family == 'all' or r['family'] == family]
        entry = {'count': len(group), 'levels': {}}
        for k in LEVELS:
            if not group:
                entry['levels'][k] = None
                continue
            stats = {metric: float(np.mean([r['levels'][k][metric] for r in group]))
                     for metric in ('hbf_mse', 'persistence_mse', 'mean_increment_mse', 'correction_mse')}
            base = stats['persistence_mse']
            stats['skill_vs_persistence'] = 1 - stats['hbf_mse'] / base if base > 1e-12 else None
            stats['win_count'] = sum(r['levels'][k]['hbf_mse'] < r['levels'][k]['persistence_mse'] for r in group)
            entry['levels'][k] = stats
        summary[family] = entry
    return summary


def run(args, config, out, report):
    import torch
    import torch_geometric
    from src.data_preparation.bre_windows import BREWindowDataset
    from src.models.hdeg.bre_hierarchy import BREHierarchy
    from src.models.hdeg.bre_semantics import load_bre_semantics
    parent = json.loads(args.training_report.read_text())
    root = args.training_report.parent
    require(parent['status'] == 'COMPLETE' and parent['protocol'] == 'bre_staged_training_v1', 'Wrong parent training report')
    require(parent['encoder_steps'] == parent['hbf_steps'] == 512, 'Parent budget differs')
    deps = {str(args.training_report): sha(args.training_report)}
    for name, digest in parent['output_hashes'].items():
        require(Path(name).name == name and sha(root / name) == digest, 'Parent artifact differs: ' + name)
        deps[str(root / name)] = digest
    # Resolve historical checkout prefixes without requiring the same absolute path.
    for old, digest in parent['source_hashes'].items():
        if old.startswith('src/'):
            path = Path(old)
        elif '/src/' in old:
            name = old.split('/src/', 1)[1]; path = Path('src') / name
        elif old.endswith('bre_semantics.yaml'):
            path = Path('configs/hdeg/bre_semantics.yaml')
        else:
            path = Path(old).name; path = Path(path)
        require(sha(path) == digest, 'Executed source differs: ' + str(path))
        deps[str(path)] = digest
    deps[str(Path(__file__).resolve())] = sha(__file__)
    checkpoint = torch.load(root / 'encoder_checkpoint.pt', map_location='cpu', weights_only=True)
    require(checkpoint['protocol'] == 'bre_reconstruction_stage_v1' and checkpoint['encoder_steps'] == 512 and checkpoint['hbf_steps'] == 0, 'Expected encoder-stage checkpoint with untouched HBF')
    require(checkpoint['selection_sha256'] == sha(root / 'selection_manifest.json'), 'Checkpoint selection differs')
    for key in ('hdeg', 'preprocessing', 'seed'):
        require(config.get(key) == checkpoint['configuration'].get(key), 'Configuration differs: ' + key)
    ds = BREWindowDataset(args.input_dir, 'train', config['preprocessing']['window_size'])
    require(ds.window_size == 30 and ds.manifest_sha256 == parent['input_manifest_sha256'], 'Input lineage/window differs')
    semantics = load_bre_semantics('configs/hdeg/bre_semantics.yaml', ds.registry)
    require(semantics == checkpoint['semantics'], 'Semantic mapping differs')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    seed = config.get('seed', 42)
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = BREHierarchy(config, semantics).to(device)
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fresh_hbf = copy.deepcopy(model.hbf.state_dict())
    old = json.loads((root / 'selection_manifest.json').read_text())
    # Replay saved trained representations before accepting the encoder.
    replay = 0.
    with np.load(root / 'update_encoder_final_representations.npz', allow_pickle=False) as saved, torch.no_grad():
        for start in range(0, 32, 8):
            indices = [r['index'] for r in old['update_pairs'][start:start + 8]]
            b = ds.get_batch(indices)
            for side in ('X', 'Y'):
                rep = model.encode(torch.from_numpy(b[side]).to(device))
                for k in LEVELS:
                    a = rep[k].cpu().numpy(); ref = saved[side + '_' + k][start:start + 8]
                    require(np.allclose(a, ref, atol=1e-6, rtol=1e-5), 'Trained encoder replay differs: ' + side + k)
                    replay = max(replay, float(np.max(np.abs(a - ref))))
    print('Encoder replay passed. Selecting matched 60-row supports...', flush=True)
    changes = np.zeros(ds.num_rows - 1, dtype=bool); previous = None
    for j, e in enumerate(ds.shards):
        a = ds.load_shard(j); v = a['values']
        require(np.all(a['status'] == 0), 'Status transitions require a separate contract')
        changes[e['start']:e['stop'] - 1] = np.any(v[1:] != v[:-1], axis=1)
        if previous is not None: changes[e['start'] - 1] = np.any(v[0] != previous)
        previous = v[-1].copy()
    cut = old['cut_row']
    train, tc = choose(changes, 0, cut, dict(quiet=512, older_change=256, future_change=256), 20261031)
    # Explicitly exclude all rows previously used for encoder fitting from probes.
    exclusions = [(r['index'], r['index'] + 29) for r in old['update_pairs']]
    probe, pc = choose(changes, cut + 60, ds.num_rows, dict(quiet=128, older_change=64, future_change=64), 20261032, exclusions)
    groups = {'update': train, 'probe': probe}
    ids_all = sorted(r['index'] for rows in groups.values() for r in rows)
    require(np.all(np.diff(ids_all) >= 60), 'Overlapping common supports')
    labels = {name: {h: classify(changes, [r['index'] for r in rows], h).tolist() for h in HORIZONS} for name, rows in groups.items()}
    write(out / 'selection_manifest.json', dict(groups=groups, horizon_labels=labels, cut_row=cut, embargo_rows=60,
        support_rows=60, candidate_counts=dict(update=tc, probe=pc), selection_rule='50/25/25 quotas classified at 30 seconds; identical X across arms',
        encoder_fit_exclusion=True, input_manifest_sha256=ds.manifest_sha256))
    rng = np.random.default_rng(20261033)
    schedule = np.concatenate([rng.permutation(1024) for _ in range(4)]).reshape(512, 8).tolist()
    write(out / 'update_schedule.json', schedule)
    # Encode once. All arms use byte-identical X representations and fixed targets.
    cache = {}
    with torch.no_grad():
        for name, rows in groups.items():
            arrays = {side: {k: [] for k in LEVELS} for side in (0,) + HORIZONS}
            for start in range(0, len(rows), 8):
                x, ys = pair_inputs(ds, [r['index'] for r in rows[start:start + 8]])
                for side, a in {0: x, **ys}.items():
                    reps = model.encode(torch.from_numpy(a).to(device))
                    for k, v in reps.items():
                        require(torch.isfinite(v).all().item(), 'Nonfinite representation')
                        arrays[side][k].append(v.cpu())
            cache[name] = {side: {k: torch.cat(v) for k, v in levels.items()} for side, levels in arrays.items()}
            np.savez_compressed(out / (name + '_representations.npz'), **{f'h{side}_{k}': v.numpy() for side, levels in cache[name].items() for k, v in levels.items()})
    variation = {}
    for name, sides in cache.items():
        variation[name] = {}
        for side, reps in sides.items():
            variation[name][str(side)] = {k: dict(mean_feature_variance=float(v.double().flatten(1).var(0, unbiased=False).mean()),
                mean_squared_norm=float(v.double().flatten(1).square().sum(1).mean())) for k, v in reps.items()}
    write(out / 'representation_variation.json', variation)
    report.update(replay_max_abs=replay, input_manifest_sha256=ds.manifest_sha256, input_metadata_hashes=ds.metadata_hashes,
                  encoder_checkpoint_sha256=sha(root / 'encoder_checkpoint.pt'), device=str(device), results={})
    for h in HORIZONS:
        arm = out / f'horizon_{h:02d}'; arm.mkdir()
        model.hbf.load_state_dict(fresh_hbf, strict=True)
        for k, v in model.hbf.state_dict().items(): require(torch.equal(v, fresh_hbf[k]), 'Arm initialization differs')
        for p in model.hbf.parameters(): p.requires_grad_(True)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        mean, scale = {}, {}
        for k in LEVELS:
            inc = cache['update'][h][k] - cache['update'][0][k]
            mean[k] = inc.mean(0).to(device)
            scale[k] = inc.double().square().mean().clamp_min(1e-12).float().to(device)
        write(arm / 'baselines.json', dict(fit_pairs=1024, persistence_scales={k: float(v) for k, v in scale.items()}, floor=1e-12))
        np.savez_compressed(arm / 'mean_increments.npz', **{k: v.cpu().numpy() for k, v in mean.items()})
        def evaluate(step):
            model.eval(); records = {}; summaries = {}; common_summaries = {}
            with torch.no_grad():
                for name, rows in groups.items():
                    entries = []
                    for start in range(0, len(rows), 8):
                        x = {k: v[start:start + 8].to(device) for k, v in cache[name][0].items()}
                        y = {k: v[start:start + 8].to(device) for k, v in cache[name][h].items()}
                        pred = model.predict(x)
                        for j, row in enumerate(rows[start:start + 8]):
                            levels = {}
                            for k in LEVELS:
                                def mse(v): return float(v.double().square().mean())
                                if step == 0: require(torch.equal(pred[k][j], x[k][j]), 'Step zero differs from persistence')
                                levels[k] = dict(hbf_mse=mse(pred[k][j] - y[k][j]), persistence_mse=mse(x[k][j] - y[k][j]),
                                    mean_increment_mse=mse(x[k][j] + mean[k] - y[k][j]), correction_mse=mse(pred[k][j] - x[k][j]))
                            entries.append(dict(index=row['index'], family=labels[name][h][start + j], common_family=row['family'], levels=levels))
                    records[name] = entries; summaries[name] = summarize(entries)
                    common_summaries[name] = summarize([dict(r, family=r['common_family']) for r in entries])
            write(arm / f'forecast_{step:04d}.json', dict(records=records, summary=summaries, common_stratum_summary=common_summaries))
            return summaries
        evaluate(0)
        optimizer = torch.optim.Adam(model.hbf.parameters(), lr=1e-4)
        history = []
        for step, ids in enumerate(schedule, 1):
            model.hbf.train(); optimizer.zero_grad(set_to_none=True)
            x = {k: v[ids].to(device) for k, v in cache['update'][0].items()}
            y = {k: v[ids].to(device) for k, v in cache['update'][h].items()}
            pred = model.predict(x)
            losses = {k: (pred[k] - y[k]).square().mean() / scale[k] for k in LEVELS}
            loss = sum(losses.values()) / 4
            require(torch.isfinite(loss).item(), 'Nonfinite forecasting loss')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.hbf.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            report['steps'][str(h)] = step
            history.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad), levels={k: float(v.detach()) for k, v in losses.items()}))
            if step == 1 or step % 64 == 0: print(f'Horizon {h}s: {step}/512 loss={float(loss.detach()):.6g}', flush=True)
        write(arm / 'history.json', history)
        summary = evaluate(512)
        for k, v in model.state_dict().items():
            require(torch.isfinite(v).all().item(), 'Nonfinite final state')
            if not k.startswith('hbf.'): require(torch.equal(v.cpu(), frozen[k]), 'Encoder changed')
        delta = float(sum((v.detach().double() - fresh_hbf[k].double()).square().sum() for k, v in model.hbf.named_parameters()).sqrt())
        torch.save(dict(protocol='bre_horizon_hbf_v1', horizon_seconds=h, hbf={k: v.detach().cpu() for k, v in model.hbf.state_dict().items()},
            encoder_checkpoint_sha256=report['encoder_checkpoint_sha256'], steps=512, selection_sha256=sha(out / 'selection_manifest.json'), configuration=config), arm / 'hbf_checkpoint.pt')
        report['results'][str(h)] = dict(summary=summary, hbf_parameter_delta_l2=delta,
            gradient_clip_fraction=float(np.mean([v['gradient_norm'] > 1 for v in history])), target_drift_mse=0.)
    ds.verify_unchanged()
    require(all(sha(path) == digest for path, digest in deps.items()), 'Source/parent changed during run')
    report.update(status='COMPLETE', source_and_parent_hashes=deps, encoder_frozen=True,
        versions=dict(torch=str(torch.__version__), torch_geometric=torch_geometric.__version__, numpy=np.__version__),
        output_hashes={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--training_report', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    with open('configs/config.yaml') as f: config = yaml.safe_load(f)
    if args.project_root_dir: config['project_root_dir'] = args.project_root_dir
    config = copy.deepcopy(config); config['preprocessing']['dataset_name'] = 'BRE'; config['preprocessing']['merge_bre_cu'] = False
    from src.utils.get_folders_utils import get_processed_folder
    out = Path(get_processed_folder(config)) / 'v2_validation' / 'bre_horizon_comparison' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', protocol='bre_horizon_comparison_v1', scientific_status='DEVELOPMENT_ONLY', steps={str(h): 0 for h in HORIZONS},
        contract=dict(horizons_seconds=HORIZONS, sampling_seconds=1, window_samples=30, update_pairs=1024, probe_pairs=256, batch_size=8, steps_per_arm=512, learning_rate=1e-4, gradient_clip=1.),
        limitations=['Train-period probes; preprocessing scaler includes probes.', 'Transition-enriched common selection is not population-frequency evaluation.',
                    'Selection uses 30-second future changes for quotas, never as model inputs.', 'No activity discrimination or anomaly-detection validation.',
                    'Raw latent MSE is not comparable across horizons; use horizon-specific persistence skill.', 'Master imputation and single-neighbor BIL limitations remain.'])
    write(out / 'configuration_snapshot.json', config)
    try: run(args, config, out, report)
    except Exception as e: report.update(status='BLOCKED', error=type(e).__name__ + ': ' + str(e))
    write(out / 'horizon_report.json', report)
    print('BRE horizon comparison:', report['status']); print('Report:', out / 'horizon_report.json')
    if report['status'] != 'COMPLETE': print(report['error']); raise SystemExit(2)


if __name__ == '__main__': main()
