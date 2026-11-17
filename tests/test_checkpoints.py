"""Bounded checks for the three supplied and hash-verified actor checkpoints."""

import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'artifacts' / 'aggregation' / 'multiseed_selected_checkpoints.csv'
with MANIFEST.open(newline='', encoding='utf-8') as stream:
    RECORDS = {int(r['training_seed']): r for r in csv.DictReader(stream)}


def checkpoint(seed):
    return ROOT / 'models' / f'seed_{seed}' / RECORDS[seed]['selected_checkpoint_file']


def inference_module():
    script = ROOT / 'scripts' / 'quick_inference.py'
    spec = importlib.util.spec_from_file_location('quick_inference_3_seeds', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('seed', [20261013, 20261014, 20261015])
def test_selected_checkpoint_sha256(seed):
    path = checkpoint(seed)
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RECORDS[seed]['selected_checkpoint_sha256']


def test_state_dict_structure_and_distinctness():
    states = {seed: torch.load(checkpoint(seed), weights_only=True, map_location='cpu')
              for seed in RECORDS}
    reference = states[20261013]
    assert len(reference) == 20
    for seed, state in states.items():
        assert list(state) == list(reference)
        assert {key: tuple(t.shape) for key, t in state.items()} == {
            key: tuple(t.shape) for key, t in reference.items()}
        assert all(torch.isfinite(t).all() for t in state.values())
    for a, b in [(20261013, 20261014), (20261013, 20261015), (20261014, 20261015)]:
        assert any(not torch.equal(states[a][k], states[b][k]) for k in reference)


@pytest.mark.parametrize('seed', [20261014, 20261015])
@pytest.mark.parametrize('n', [20, 50, 100])
def test_new_checkpoint_greedy_cpu(seed, n):
    xy = np.random.default_rng(8302026 + n).random((n, 2))
    tour, length = inference_module().predict(xy, checkpoint(seed), 'greedy',
                                               seed=8302026 + n, device='cpu')
    assert sorted(tour.tolist()) == list(range(n))
    assert np.isfinite(length) and length > 0


@pytest.mark.parametrize('seed', [20261014, 20261015])
def test_new_checkpoint_sampling_smoke_cpu(seed):
    # Three draws exercise the sampling path; this is NOT the official Best-of-512 benchmark.
    xy = np.random.default_rng(8302046).random((20, 2))
    tour, length = inference_module().predict(xy, checkpoint(seed), 'best-of-512',
                                               sample_count=3, seed=8302046, device='cpu')
    assert sorted(tour.tolist()) == list(range(20))
    assert np.isfinite(length) and length > 0
