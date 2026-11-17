from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code_snapshot'))
sys.path.insert(0, str(ROOT / 'src'))
from env import Env_tsp
from tsp_drl.algorithms.reference_baselines import (
    tour_length_np, distance_matrix, nearest_neighbor, two_opt, ant_colony_tsp)


def is_permutation(tour, n):
    return sorted(map(int, tour)) == list(range(n))


def sample_coords(n=6):
    return np.random.default_rng(777).random((n, 2))


def test_closed_tour_length():
    xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
    tour=np.array([0,1,2])
    expect=2+2**0.5
    assert abs(tour_length_np(xy,tour)-expect)<1e-12
    env=Env_tsp(type('Cfg',(),dict(batch=1,city_t=3))())
    actual=env.stack_l_fast(torch.tensor(xy[None],dtype=torch.float32), torch.tensor(tour[None]))[0]
    assert abs(float(actual)-expect)<1e-6


def test_classical_solvers_are_valid_and_costs_consistent():
    xy=sample_coords(7)
    nn=nearest_neighbor(xy)
    opt=two_opt(xy,nn)
    aco=ant_colony_tsp(xy,seed=101,n_ants=4,n_iter=3)
    for tour in (nn,opt,aco):
        assert is_permutation(tour,len(xy))
        assert np.isfinite(tour_length_np(xy,tour))
    assert tour_length_np(xy,opt) <= tour_length_np(xy,nn)+1e-10
    assert np.allclose(distance_matrix(xy),distance_matrix(xy).T)


def test_extracted_baselines_match_notebook_source():
    """Run notebook's original function definitions on identical inputs, without Colab cells."""
    import ast
    p=ROOT/'notebooks/seed_20261015_evaluation_corrected_torch2110_v1_2.ipynb'
    nb=json.loads(p.read_text())
    code=''.join([c for c in nb['cells'] if c['cell_type']=='code'][1]['source'])
    original=ast.parse(code)
    names={'tour_length_np','distance_matrix','nearest_neighbor','two_opt','ant_colony_tsp'}
    selected=[]
    for node in original.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Tuple) and any(isinstance(e,ast.Name) and e.id=='ACO_ANTS' for e in t.elts) for t in node.targets):
            selected.append(node)
    ns={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(p),'exec'),ns)
    xy=sample_coords(7)
    for fn_name,args in [('nearest_neighbor',(xy,)),('two_opt',(xy,nearest_neighbor(xy))),('ant_colony_tsp',(xy,101))]:
        fn=globals()[fn_name]
        if fn_name=='ant_colony_tsp':
            # full default config is the same exact implementation; use small settings for speed.
            assert np.array_equal(fn(*args,n_ants=4,n_iter=3),ns[fn_name](*args,n_ants=4,n_iter=3))
        else:
            assert np.array_equal(fn(*args),ns[fn_name](*args))


def test_policy_gradient_and_baseline_detach_semantics():
    # Exact math of code_snapshot/train.py; verifies gradient routes, not a training rerun.
    predicted=torch.tensor([3.0,4.0],requires_grad=True)
    real=torch.tensor([4.0,6.0])
    log_likelihood=torch.tensor([-2.0,-1.0],requires_grad=True)
    baseline_loss=torch.nn.MSELoss()(predicted,real.detach())
    baseline_loss.backward()
    assert predicted.grad is not None
    predicted.grad.zero_()
    adv=real.detach()-predicted.detach()
    actor_loss=(adv*log_likelihood).mean()
    actor_loss.backward()
    assert torch.allclose(log_likelihood.grad,adv/2)
    assert torch.allclose(predicted.grad,torch.zeros_like(predicted.grad))


@pytest.mark.parametrize('n',[20,50,100])
def test_real_checkpoint_greedy_cpu(n):
    script=ROOT/'scripts/quick_inference.py'
    spec=importlib.util.spec_from_file_location('quick_inference',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    coords=sample_coords(n)
    tour,cost=module.predict(coords,module.DEFAULT_CHECKPOINT,'greedy',seed=123,device='cpu')
    assert is_permutation(tour,n) and cost>0


def test_real_checkpoint_sampling_smoke_cpu():
    script=ROOT/'scripts/quick_inference.py'
    spec=importlib.util.spec_from_file_location('quick_inference',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    coords=sample_coords(20)
    tour,cost=module.predict(coords,module.DEFAULT_CHECKPOINT,'best-of-512',sample_count=3,seed=123,device='cpu')
    assert is_permutation(tour,20) and cost>0
