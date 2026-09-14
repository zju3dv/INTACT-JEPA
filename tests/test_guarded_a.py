import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import pytest
import torch
from omegaconf import OmegaConf

from abcd_solvers import GuardedCEMSolver
from cem_solvers import PureCEMSolver
from clear_eval import explicit_cem_solver, finalize_clear_result
from evaluation_contract import (
    CLEAR_SOLVER_TARGETS,
    GUARDED_A_CONFIG,
    MODE_CONTRACTS,
    validate_guarded_a_config,
)


def _make_solver(**overrides):
    options = {
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
        "trust_lambda": 0.0,
    }
    options.update(overrides)
    return GuardedCEMSolver(**options)


def test_guarded_a_matches_release_contract():
    solver = _make_solver()
    assert solver.evaluation_mode == "guarded_a"
    assert solver.num_samples == 128
    assert solver.n_steps == 3
    assert solver.topk == 16
    assert solver.var_scale == 0.25
    assert MODE_CONTRACTS["guarded_a"]["search_budget"] == "128-samples-x-3-rounds"
    assert MODE_CONTRACTS["guarded_a"]["direct_reference_preserved"] == "true"
    assert MODE_CONTRACTS["guarded_a"]["planning_horizon"] == "5"
    assert MODE_CONTRACTS["guarded_a"]["receding_horizon"] == "5"


def test_guarded_a_rejects_a_mislabeled_horizon():
    observed = dict(GUARDED_A_CONFIG)
    observed["receding_horizon"] = 1
    with pytest.raises(ValueError, match="receding_horizon"):
        validate_guarded_a_config(observed)


def test_guarded_a_rejects_noncanonical_options():
    with pytest.raises(ValueError, match="actor_covariance=false"):
        _make_solver(actor_covariance=True)
    with pytest.raises(ValueError, match="trust_lambda=0"):
        _make_solver(trust_lambda=0.1)


def test_pure_cem_reports_sequences_and_action_steps_separately(monkeypatch):
    solver = PureCEMSolver(
        model=torch.nn.Linear(1, 1),
        batch_size=2,
        num_samples=300,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device="cpu",
        seed=0,
    )
    solver._config = type("Plan", (), {"horizon": 5, "action_block": 1})()
    solver._action_dim = 2

    monkeypatch.setattr(
        "cem_solvers.CEMSolver.solve",
        lambda self, info_dict, init_action=None: {"actions": init_action},
    )
    result = solver.solve({"pixels": torch.zeros(3, 1, 1)})
    timing = result["timing"]
    assert timing["candidate_action_sequences"] == 3 * 300 * 30
    assert timing["candidate_action_steps"] == 3 * 300 * 30 * 5
    assert timing["configured_candidate_sequences_per_solve"] == 9_000
    assert timing["configured_candidate_action_steps_per_solve"] == 45_000


def test_clear_adapter_installs_canonical_guarded_configuration():
    class _Runner:
        @staticmethod
        def _compose_config(*args, **kwargs):
            return OmegaConf.create(
                {
                    "plan_config": {
                        "horizon": 1,
                        "receding_horizon": 1,
                        "action_block": 5,
                    },
                    "solver": {
                        "_target_": "upstream.CEMSolver",
                        "num_samples": 30,
                        "var_scale": 1.0,
                        "n_steps": 10,
                        "topk": 3,
                    }
                }
            )

    original = _Runner._compose_config
    with explicit_cem_solver(_Runner, "guarded_a"):
        solver = _Runner._compose_config().solver
        assert solver._target_ == CLEAR_SOLVER_TARGETS["guarded_a"]
        assert solver.num_samples == 128
        assert solver.n_steps == 3
        assert solver.topk == 16
        assert solver.var_scale == 0.25
        cfg = _Runner._compose_config()
        assert cfg.plan_config.horizon == 5
        assert cfg.plan_config.receding_horizon == 5
    assert _Runner._compose_config is original


def test_clear_result_records_guarded_budget_identity():
    result = {
        "inference": {"solver_target": CLEAR_SOLVER_TARGETS["guarded_a"]},
        "solver": {},
    }
    finalized = finalize_clear_result(result, "guarded_a")
    assert finalized["solver"]["identity"] == "Guarded A"
    assert finalized["solver"]["candidate_sequences_per_solve"] == 384
    assert finalized["solver"]["candidate_action_steps_per_solve"] == 1920
    assert finalized["solver"]["final_mean_rescores_per_solve"] == 1
    assert finalized["solver"]["horizon"] == 5
    assert finalized["solver"]["receding_horizon"] == 5
    assert finalized["action_history_protocol"]["target_action_visible"] is False


def test_frozen_guarded_results_have_complete_distinct_cohort_identity():
    root = Path(__file__).parents[1]
    path = root / "results" / "guarded_a_official.json"
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["protocol"]["name"] == "official_expert_continuation_v1"
    assert result["solver"]["sdpa_backend"] == "math"
    assert result["solver"]["sampled_sequences_per_solve"] == 384
    assert result["solver"]["sampled_action_steps_per_solve"] == 1920
    assert result["provenance"]["audited_cell_count"] == 72
    assert result["provenance"]["stable_worldmodel_world_sha256"] == result[
        "provenance"
    ]["official_evaluator_sha256"]
    assert set(
        result["provenance"]["evaluated_eval_adapter_sha256_by_cohort"]
    ) == {"task_specific_e1", "shared_encoder_e5"}
    for key in (
        "official_evaluator_sha256",
        "stable_worldmodel_world_sha256",
        "evaluated_history_policy_sha256",
        "evaluated_guarded_solver_sha256",
    ):
        assert len(result["provenance"][key]) == 64
    for digest in result["provenance"][
        "evaluated_eval_adapter_sha256_by_cohort"
    ].values():
        assert len(digest) == 64
    release_solver_hash = hashlib.sha256(
        (root / "abcd_solvers.py").read_bytes()
    ).hexdigest()
    assert (
        result["provenance"]["release_guarded_solver_sha256"]
        == release_solver_hash
    )

    cohorts = {cohort["id"]: cohort for cohort in result["cohorts"]}
    assert cohorts["task_specific_e1"]["training_seeds"] == [3072, 3073, 3074]
    assert cohorts["shared_encoder_e5"]["training_seeds"] == [0, 42, 3072]
    assert cohorts["task_specific_e1"]["aggregate_mean_sample_std_percent"][
        "macro"
    ] == pytest.approx([96.58333333333333, 0.4409585518440948])
    assert cohorts["shared_encoder_e5"]["aggregate_mean_sample_std_percent"][
        "macro"
    ] == pytest.approx([90.52777777777777, 0.3154949081000217])

    audited_cells = 0
    for cohort in cohorts.values():
        task_seed_scores = {}
        for task, seed_rows in cohort["success_rate_percent_by_eval_seed"].items():
            task_seed_scores[task] = {}
            for seed, values in seed_rows.items():
                assert len(values) == len(result["statistics"]["evaluation_seeds"])
                pooled = mean(values)
                task_seed_scores[task][seed] = pooled
                assert cohort["pooled_success_rate_percent_by_training_seed"][task][
                    seed
                ] == pytest.approx(pooled)
                assert len(cohort["checkpoint_sha256"][task][seed]) == 64
                audited_cells += len(values)

        for task, seed_scores in task_seed_scores.items():
            aggregate = cohort["aggregate_mean_sample_std_percent"][task]
            assert aggregate == pytest.approx(
                [mean(seed_scores.values()), stdev(seed_scores.values())]
            )

        macros = [
            mean(task_seed_scores[task][str(seed)] for task in task_seed_scores)
            for seed in cohort["training_seeds"]
        ]
        assert cohort["aggregate_mean_sample_std_percent"]["macro"] == pytest.approx(
            [mean(macros), stdev(macros)]
        )

    assert audited_cells == result["provenance"]["audited_cell_count"] == 72
