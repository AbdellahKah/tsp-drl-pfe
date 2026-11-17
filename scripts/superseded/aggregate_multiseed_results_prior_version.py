#!/usr/bin/env python3
"""Aggregate three independent TSP DRL training-seed evaluations.

Default root is the Google Drive project root used by the Colab notebooks.
No model training is performed. Existing input evaluations are read only.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_EVALUATIONS = [
    ('final_evaluation_v1_1', 20261013),
    ('final_evaluation_seed20261014_v1_2', 20261014),
    ('final_evaluation_seed20261015_v1_2', 20261015),
]
CITY_SIZES = [20, 50, 100]
N_INSTANCES = 100
BOOTSTRAP_REPLICATES = 10000
BASELINE_METHODS = ['NN_start0', 'NN_start0_plus_2opt', 'ACO_fixed_parameters']
DRL_METHODS = ['DRL_greedy', 'DRL_best_of_512']


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def get_training_run_id(manifest: dict) -> str:
    run_id = manifest.get('training_run_id') or manifest.get('pilot_run_id')
    if not run_id:
        raise ValueError('Evaluation manifest has neither training_run_id nor legacy pilot_run_id.')
    return str(run_id)


def get_seed(run_id: str, manifest: dict, project_root: Path) -> int:
    seed = manifest.get('training_seed')
    if seed is None:
        m = re.search(r'seed(\d+)', run_id)
        if m:
            seed = int(m.group(1))
    if seed is None:
        pre_path = project_root / 'runs' / run_id / 'manifest_pre.json'
        if pre_path.is_file():
            seed = load_json(pre_path).get('train_seed')
    if seed is None:
        raise ValueError(f'Cannot determine training seed for {run_id}.')
    return int(seed)


def check_coords_consistent(eval_paths):
    baseline_validation = None
    baseline_final = None
    final_hash_map = None
    import hashlib
    for eval_dir, run_id, seed in eval_paths:
        val_path = eval_dir / 'validation' / 'validation_n20_seed7302026_count100.npz'
        if not val_path.is_file():
            raise FileNotFoundError(f'Missing validation coordinates: {val_path}')
        with np.load(val_path) as z:
            val_coords = z['coords'].copy()
        if val_coords.shape != (N_INSTANCES, 20, 2):
            raise ValueError(f'Unexpected validation shape for {run_id}: {val_coords.shape}')
        if baseline_validation is None:
            baseline_validation = val_coords
        elif not np.array_equal(baseline_validation, val_coords):
            raise ValueError(f'Validation coordinates differ across training runs; cannot aggregate under the paired protocol ({run_id}).')

        inst_dir = eval_dir / 'final_test_instances'
        instance_manifest = load_json(inst_dir / 'final_test_instances_manifest.json')
        current_arrays = {}
        current_hash_map = {}
        for n in CITY_SIZES:
            info = instance_manifest['city_sizes'][str(n)]
            npz_path = inst_dir / info['file']
            with np.load(npz_path) as z:
                arr = z['coords'].copy()
                stored_seed = int(z['seed'])
            if arr.shape != (N_INSTANCES, n, 2):
                raise ValueError(f'Unexpected final-test shape in {npz_path}: {arr.shape}')
            if stored_seed != 8302026 + n:
                raise ValueError(f'Unexpected final-test effective seed in {npz_path}: {stored_seed}')
            current_arrays[n] = arr
            # Hash the canonical coordinate array, not the compressed NPZ container (whose metadata may vary).
            canonical = np.ascontiguousarray(arr)
            current_hash_map[n] = hashlib.sha256(canonical.tobytes()).hexdigest()
        if final_hash_map is None:
            final_hash_map = current_hash_map
        elif current_hash_map != final_hash_map:
            raise ValueError(f'Final-test coordinate arrays differ across runs. Cannot aggregate paired results ({run_id}).')
        if baseline_final is None:
            baseline_final = current_arrays
        else:
            for n in CITY_SIZES:
                if not np.array_equal(baseline_final[n], current_arrays[n]):
                    raise ValueError(f'Final-test coordinates differ for n={n}, run={run_id}.')
    return final_hash_map


def validate_raw_rows(df, coords_by_n, run_label):
    required = {'n', 'instance_id', 'method', 'tour_length', 'tour_indices'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{run_label} results missing columns {sorted(missing)}')
    if not np.isfinite(pd.to_numeric(df['tour_length'], errors='coerce')).all():
        raise ValueError(f'Non-finite tour lengths in {run_label}')
    for r in df.itertuples(index=False):
        n, iid = int(r.n), int(r.instance_id)
        if n not in coords_by_n or not (0 <= iid < N_INSTANCES):
            raise ValueError(f'Invalid test key in {run_label}: n={n}, instance={iid}')
        tour = np.asarray([int(x) for x in str(r.tour_indices).split(',')], dtype=np.int64)
        if sorted(tour.tolist()) != list(range(n)):
            raise ValueError(f'Invalid tour permutation in {run_label}: n={n}, instance={iid}, method={r.method}')
        ordered = coords_by_n[n][iid][tour].astype(np.float64)
        measured = float(np.linalg.norm(ordered[1:] - ordered[:-1], axis=1).sum() + np.linalg.norm(ordered[0] - ordered[-1]))
        if abs(measured - float(r.tour_length)) > 1e-6:
            raise ValueError(f'Tour-length recomputation mismatch in {run_label}: n={n}, instance={iid}, method={r.method}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/content/drive/MyDrive/PFE_TSP_DRL'),
                        help='Project root containing runs/ and final_evaluation_* folders.')
    parser.add_argument('--output-folder', default='final_evaluation_multi_seed_v1_2',
                        help='New output folder name under --project-root.')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing the aggregation output folder only; inputs remain read-only.')
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_root = project_root / args.output_folder
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f'{output_root} already exists. Use a fresh --output-folder or pass --overwrite to replace aggregation outputs.')

    eval_items = []
    eval_manifest_records = []
    environment_records = []
    for folder_name, expected_seed in EXPECTED_EVALUATIONS:
        eval_dir = project_root / folder_name
        eval_manifest = load_json(eval_dir / 'final_evaluation_manifest.json')
        run_id = get_training_run_id(eval_manifest)
        actual_seed = get_seed(run_id, eval_manifest, project_root)
        if actual_seed != expected_seed:
            raise ValueError(f'{folder_name}: expected training seed {expected_seed}, found {actual_seed} ({run_id}).')
        selection = load_json(eval_dir / 'checkpoint_selection.json')
        if selection.get('selected_checkpoint_file') != eval_manifest.get('selected_checkpoint_file'):
            raise ValueError(f'Selected checkpoint mismatch between manifest and selection file in {folder_name}.')
        if not (eval_dir / 'final_test_per_instance.csv').is_file():
            raise FileNotFoundError(f'Missing final_test_per_instance.csv in {eval_dir}.')
        eval_items.append((eval_dir, run_id, actual_seed))
        environment_records.append({
            'evaluation_folder': folder_name,
            'python': eval_manifest.get('python'),
            'pytorch': eval_manifest.get('pytorch'),
            'cuda_runtime': eval_manifest.get('cuda_runtime'),
            'gpu': eval_manifest.get('gpu'),
        })
        eval_manifest_records.append({
            'evaluation_folder': folder_name,
            'training_run_id': run_id,
            'training_seed': actual_seed,
            'selected_checkpoint_file': selection.get('selected_checkpoint_file'),
            'selected_checkpoint_sha256': eval_manifest.get('selected_checkpoint_sha256'),
            'completed_optimizer_updates': selection.get('completed_optimizer_updates'),
        })

    # Strictly require a common inference stack: sampled decoding can depend on CUDA/PyTorch behavior.
    for key in ['pytorch', 'cuda_runtime', 'gpu']:
        vals = {str(record.get(key)) for record in environment_records}
        if len(vals) != 1:
            raise ValueError(f'Inference environment differs across evaluation folders for {key}: {sorted(vals)}. Re-run evaluations in a matching runtime or report the environment difference separately.')
    python_versions = {str(record.get('python')) for record in environment_records}
    if len(python_versions) != 1:
        print('WARNING: Python versions differ across evaluations:', sorted(python_versions))
    # The new evaluation folders must use the same exact validation and final-test coordinates.
    hash_map = check_coords_consistent(eval_items)
    # Load coordinate arrays once for independent route-length validation.
    first_eval = eval_items[0][0]
    coords_by_n = {}
    first_manifest = load_json(first_eval / 'final_test_instances' / 'final_test_instances_manifest.json')
    for n in CITY_SIZES:
        info = first_manifest['city_sizes'][str(n)]
        with np.load(first_eval / 'final_test_instances' / info['file']) as z:
            coords_by_n[n] = z['coords'].copy()

    frames = []
    baseline_reference = None
    baseline_compare = None
    for index, (eval_dir, run_id, seed) in enumerate(eval_items):
        df = pd.read_csv(eval_dir / 'final_test_per_instance.csv')
        validate_raw_rows(df, coords_by_n, eval_dir.name)
        baselines = df[df['method'].isin(BASELINE_METHODS)].copy()
        drl = df[df['method'].isin(DRL_METHODS)].copy()
        if set(baselines['method'].unique()) != set(BASELINE_METHODS):
            raise ValueError(f'Baseline method set incomplete in {eval_dir.name}.')
        if set(drl['method'].unique()) != set(DRL_METHODS):
            raise ValueError(f'DRL method set incomplete in {eval_dir.name}.')
        if len(drl) != len(CITY_SIZES) * N_INSTANCES * len(DRL_METHODS):
            raise ValueError(f'Expected {len(CITY_SIZES) * N_INSTANCES * len(DRL_METHODS)} DRL rows in {eval_dir.name}; found {len(drl)}.')
        compare = baselines[['n', 'instance_id', 'method', 'tour_length', 'tour_indices']].sort_values(['n','instance_id','method']).reset_index(drop=True)
        if baseline_compare is None:
            baseline_compare = compare
            baseline_reference = baselines.copy()
        else:
            joined = baseline_compare.merge(compare, on=['n','instance_id','method'], suffixes=('_reference','_current'), validate='one_to_one')
            if len(joined) != len(baseline_compare) or not np.allclose(joined['tour_length_reference'], joined['tour_length_current'], rtol=0, atol=1e-9):
                raise ValueError(f'Classical baseline lengths differ across evaluation folders ({eval_dir.name}); inspect seeds and solver configuration.')
            if not (joined['tour_indices_reference'].astype(str).to_numpy() == joined['tour_indices_current'].astype(str).to_numpy()).all():
                raise ValueError(f'Classical baseline tours differ across evaluation folders ({eval_dir.name}).')
        drl['run_id'] = run_id
        drl['training_seed'] = seed
        drl['evaluation_folder'] = eval_dir.name
        frames.append(drl)
        print(f'Validated {eval_dir.name}: seed={seed}; DRL rows={len(drl)}; baseline rows={len(baselines)}')

    # Keep only one shared copy of each deterministic baseline; retain DRL rows from all seeds.
    baseline_reference['run_id'] = 'shared_baselines'
    baseline_reference['training_seed'] = np.nan
    baseline_reference['evaluation_folder'] = eval_items[0][0].name
    combined = pd.concat([baseline_reference, *frames], ignore_index=True, sort=False)
    combined['n'] = combined['n'].astype(int)
    combined['instance_id'] = combined['instance_id'].astype(int)
    combined['tour_length'] = pd.to_numeric(combined['tour_length'], errors='raise')
    if combined.duplicated(['n', 'instance_id', 'method', 'run_id']).any():
        raise ValueError('Duplicate (n, instance, method, run_id) rows in combined data.')

    reference = combined.groupby(['n', 'instance_id'])['tour_length'].min().rename('best_observed_reference')
    combined = combined.join(reference, on=['n', 'instance_id'])
    combined['gap_vs_best_observed_percent'] = 100.0 * (combined['tour_length'] / combined['best_observed_reference'] - 1.0)
    combined['gap_vs_best_observed_percent'] = combined['gap_vs_best_observed_percent'].clip(lower=0.0)
    # All inputs and consistency checks have passed. Only now create/replace aggregation outputs.
    if output_root.exists():
        import shutil
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    combined_path = output_root / 'multiseed_combined_per_instance.csv'
    combined.to_csv(combined_path, index=False)

    per_seed = (combined.groupby(['n','run_id','training_seed','method'], dropna=False)
                .agg(instances=('instance_id','nunique'), mean_tour_length=('tour_length','mean'),
                     sd_tour_length_across_instances=('tour_length','std'),
                     median_tour_length=('tour_length','median'),
                     mean_gap_vs_best_observed_percent=('gap_vs_best_observed_percent','mean'),
                     sd_gap_vs_best_observed_percent=('gap_vs_best_observed_percent','std'))
                .reset_index())
    per_seed_path = output_root / 'multiseed_per_seed_summary.csv'
    per_seed.to_csv(per_seed_path, index=False)

    across_rows = []
    for n in CITY_SIZES:
        for method in DRL_METHODS:
            sub = per_seed[(per_seed['n'] == n) & (per_seed['method'] == method) & (per_seed['run_id'] != 'shared_baselines')]
            seed_means = sub['mean_tour_length'].to_numpy(dtype=float)
            across_rows.append({
                'n': n, 'method': method, 'n_training_seeds': len(seed_means),
                'mean_of_seed_level_means': float(seed_means.mean()) if len(seed_means) else np.nan,
                'sd_between_training_seed_means': float(seed_means.std(ddof=1)) if len(seed_means) > 1 else np.nan,
                'minimum_seed_level_mean': float(seed_means.min()) if len(seed_means) else np.nan,
                'maximum_seed_level_mean': float(seed_means.max()) if len(seed_means) else np.nan,
                'interpretation': 'descriptive variability across independently trained models; only three seeds'
            })
        for method in BASELINE_METHODS:
            row = per_seed[(per_seed['n'] == n) & (per_seed['method'] == method) & (per_seed['run_id'] == 'shared_baselines')]
            if len(row) != 1:
                raise ValueError(f'Expected one shared-baseline summary for n={n}, method={method}.')
            x = row.iloc[0]
            across_rows.append({
                'n': n, 'method': method, 'n_training_seeds': 0,
                'mean_of_seed_level_means': float(x['mean_tour_length']),
                'sd_between_training_seed_means': np.nan,
                'minimum_seed_level_mean': float(x['mean_tour_length']),
                'maximum_seed_level_mean': float(x['mean_tour_length']),
                'interpretation': 'single shared baseline evaluated on the same 100 instances; SD shown in per-seed summary is across instances'
            })
    across = pd.DataFrame(across_rows)
    across_path = output_root / 'multiseed_across_seed_summary.csv'
    across.to_csv(across_path, index=False)

    # Paired bootstrap intervals for each seed, using the same instance IDs for DRL and baselines.
    rng = np.random.default_rng(8802026)
    paired_rows = []
    for eval_dir, run_id, seed in eval_items:
        seed_df = combined[(combined['run_id'].isin(['shared_baselines', run_id])) & (combined['n'].isin(CITY_SIZES))]
        for n in CITY_SIZES:
            wide = seed_df[seed_df['n'] == n].pivot(index='instance_id', columns='method', values='tour_length').sort_index()
            expected_cols = set(BASELINE_METHODS + DRL_METHODS)
            if set(wide.columns) != expected_cols or len(wide) != N_INSTANCES:
                raise ValueError(f'Incomplete paired comparison matrix for seed {seed}, n={n}.')
            for drl_method in DRL_METHODS:
                for baseline_method in BASELINE_METHODS:
                    delta = (wide[drl_method] - wide[baseline_method]).to_numpy(dtype=float)
                    boot_means = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
                    for b in range(BOOTSTRAP_REPLICATES):
                        boot_means[b] = rng.choice(delta, size=len(delta), replace=True).mean()
                    lo, hi = np.quantile(boot_means, [0.025, 0.975])
                    paired_rows.append({
                        'training_seed': seed, 'run_id': run_id, 'n': n,
                        'drl_method': drl_method, 'baseline_method': baseline_method,
                        'paired_mean_length_difference_drl_minus_baseline': float(delta.mean()),
                        'bootstrap_95pct_ci_low': float(lo), 'bootstrap_95pct_ci_high': float(hi),
                        'instances': len(delta), 'bootstrap_replicates': BOOTSTRAP_REPLICATES,
                        'interpretation': 'negative favors DRL in mean length; conditional on this model and test set; descriptive; no multiplicity correction'
                    })
    paired = pd.DataFrame(paired_rows)
    paired_path = output_root / 'multiseed_paired_bootstrap_by_seed.csv'
    paired.to_csv(paired_path, index=False)

    selected_path = output_root / 'multiseed_selected_checkpoints.csv'
    pd.DataFrame(eval_manifest_records).to_csv(selected_path, index=False)
    manifest_out = {
        'aggregation_id': args.output_folder,
        'created_at_local': pd.Timestamp.now().isoformat(),
        'project_root': str(project_root),
        'training_runs': eval_manifest_records,
        'validation_seed': 7302026,
        'final_test_base_seed': 8302026,
        'final_test_effective_seed_rule': '8302026 + n',
        'instances_per_size': N_INSTANCES,
        'city_sizes': CITY_SIZES,
        'final_test_coordinate_sha256_by_n': {str(k): v for k, v in hash_map.items()},
        'training_seeds': [x[2] for x in eval_items],
        'inference_environment': environment_records,
        'bootstrap_replicates': BOOTSTRAP_REPLICATES,
        'notes': [
            'Validation and final-test coordinate arrays were checked to be identical across all three evaluation folders.',
            'Classical baseline tours and lengths were checked for exact agreement across folders and included only once.',
            'Best-observed reference is the minimum across all included baseline methods, seed-specific DRL checkpoints and decoding modes; it is not an exact optimum.',
            'Between-seed standard deviation is the sample SD of three seed-level mean tour lengths; descriptive, not a reliable population variance estimate.',
            'Paired bootstrap intervals quantify instance-level uncertainty conditional on each fixed trained model; they do not replace between-seed analysis.'
        ],
        'outputs': [combined_path.name, per_seed_path.name, across_path.name, paired_path.name, selected_path.name]
    }
    manifest_path = output_root / 'multiseed_aggregation_manifest.json'
    manifest_path.write_text(json.dumps(manifest_out, indent=2), encoding='utf-8')
    print('\nAGGREGATION COMPLETE')
    print('Output folder:', output_root)
    print('\nAcross-seed means (between-seed SD is not test-instance SD):')
    print(across.to_string(index=False))
    print('\nSaved files:')
    for path in [combined_path, per_seed_path, across_path, paired_path, selected_path, manifest_path]:
        print(' ', path)

if __name__ == '__main__':
    main()
