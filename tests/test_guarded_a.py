import pytest
import torch

from abcd_solvers import GuardedCEMSolver


def make_solver(**overrides):
    kwargs = {
        "model": torch.nn.Linear(1, 1),
        "batch_size": 1,
        "num_samples": 128,
        "var_scale": 0.25,
        "n_steps": 3,
        "topk": 16,
        "device": "cpu",
        "seed": 0,
        "update_alpha": 1.0,
        "std_floor": 1e-4,
        "std_cap": 10.0,
        "actor_covariance": False,
        "actor_cov_target_rms": 0.25,
        "actor_cov_min": 0.05,
        "actor_cov_max": 0.5,
        "trust_lambda": 0.0,
    }
    kwargs.update(overrides)
    return GuardedCEMSolver(**kwargs)


def test_guarded_a_matches_paper_contract():
    solver = make_solver()
    assert solver.evaluation_mode == "guarded_a"
    assert solver.num_samples == 128
    assert solver.n_steps == 3
    assert solver.topk == 16
    assert solver.var_scale == 0.25


def test_guarded_a_rejects_noncanonical_abcd_options():
    with pytest.raises(ValueError, match="actor_covariance=false"):
        make_solver(actor_covariance=True)
    with pytest.raises(ValueError, match="trust_lambda=0"):
        make_solver(trust_lambda=0.1)
