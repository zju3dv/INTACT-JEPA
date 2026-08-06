from pathlib import Path

from hydra import compose, initialize_config_dir


def test_all_paper_configs_compose():
    expected = {
        "lewm": ("lewm", 0.0, 0.0, None),
        "intact_inverse": ("inverse", 0.1, 0.0, "four_slot"),
        "intact_goal_only": ("goal_only", 0.0, 0.05, "four_slot"),
        "intact_goal": ("displacement", 0.1, 0.05, "four_slot"),
        "intact_waypoint": ("waypoint", 0.1, 0.05, "four_slot"),
    }
    config_dir = str(Path(__file__).parents[1] / "config" / "train")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        for config_name, (variant, local, goal, layout) in expected.items():
            for task in ("pusht", "ogb", "dmc", "tworoom"):
                cfg = compose(config_name=config_name, overrides=[f"data={task}"])
                assert cfg.data.dataset.num_steps == 8
                assert cfg.model.predict_residual is False
                assert cfg.variant == variant
                assert cfg.loss.intent.local_weight == local
                assert cfg.loss.intent.goal_weight == goal
                assert cfg.loss.intent.goal_start == 0
                assert cfg.loss.intent.local_start == 0
                if layout is None:
                    assert cfg.model.intent_actor is None
                else:
                    assert cfg.model.intent_actor.feature_layout == layout
                assert cfg.loss.sigreg.weight == 0.02


def test_eval_modes_compose():
    config_dir = str(Path(__file__).parents[1] / "config" / "eval")
    expected = {
        "direct": "direct_solver.DirectSolver",
        "pure_cem": "cem_solvers.PureCEMSolver",
        "actor_cem": "cem_solvers.ActorCEMSolver",
        "guarded_a": "abcd_solvers.GuardedCEMSolver",
    }
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        for mode, target in expected.items():
            cfg = compose(config_name="pusht", overrides=[f"solver={mode}"])
            assert cfg.solver._target_ == target
            if mode == "guarded_a":
                assert cfg.solver.num_samples == 128
                assert cfg.solver.n_steps == 3
                assert cfg.solver.topk == 16
                assert cfg.solver.var_scale == 0.25
