"""Consistency checks using archived data, not neural-network re-evaluation."""
from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tsp_drl.algorithms.reference_baselines import tour_length_np


def test_seed13_selected_checkpoint_sha256():
    p = ROOT / 'models/seed_20261013/train20_1013_17_14_step12999_act.pt'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '87610c76c1239c32d2bdba584196a00af85a45d008d8ba3fed181a31891ee5b4'


def test_all_recorded_tours_consistent_with_regenerated_test_coordinates():
    df = pd.read_csv(ROOT / 'artifacts/aggregation/multiseed_combined_per_instance.csv')
    assert len(df) == 2700
    assert set(df.n.unique()) == {20, 50, 100}
    coords = {n: np.random.default_rng(8302026 + n).random((100, n, 2), dtype=np.float64).astype(np.float32)
              for n in (20, 50, 100)}
    for row in df.itertuples(index=False):
        tour = np.fromstring(row.tour_indices, sep=',', dtype=np.int64)
        assert np.array_equal(np.sort(tour), np.arange(row.n))
        cost = tour_length_np(coords[row.n][row.instance_id], tour)
        assert abs(cost - row.tour_length) < 1e-6
    ref = df.groupby(['n','instance_id']).tour_length.min()
    for row in df.itertuples(index=False):
        assert abs(ref.loc[row.n, row.instance_id] - row.best_observed_reference) < 1e-10
        gap = max(0.0, 100 * (row.tour_length / row.best_observed_reference - 1))
        assert abs(gap - row.gap_vs_best_observed_percent) < 1e-8


def test_archived_multiseed_mean_and_sample_sd():
    d = pd.read_csv(ROOT / 'artifacts/aggregation/multiseed_combined_per_instance.csv')
    archived = pd.read_csv(ROOT / 'artifacts/aggregation/multiseed_across_seed_summary.csv')
    drl = d[d.method.str.startswith('DRL')]
    seed_means = drl.groupby(['n','method','training_seed']).tour_length.mean()
    derived = seed_means.groupby(['n','method']).agg(['mean','std'])
    for r in archived[archived.n_training_seeds == 3].itertuples(index=False):
        ours = derived.loc[r.n, r.method]
        assert abs(ours['mean'] - r.mean_of_seed_level_means) < 1e-11
        assert abs(ours['std'] - r.sd_between_training_seed_means) < 1e-11
