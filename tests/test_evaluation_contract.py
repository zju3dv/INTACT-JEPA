from evaluation_contract import (
    CLEAR_SOLVER_TARGETS,
    OFFICIAL_SOLVER_TARGETS,
    build_evaluation_contract,
)


def test_all_released_solver_modes_have_explicit_protocol_contracts():
    for protocol, targets in (
        ("official", OFFICIAL_SOLVER_TARGETS),
        ("clear-lewm-v0.8", CLEAR_SOLVER_TARGETS),
    ):
        for mode, target in targets.items():
            contract = build_evaluation_contract(protocol, mode, target)
            assert contract["protocol"] == protocol
            assert contract["inference_mode"] == mode
            assert contract["solver_target"] == target


def test_contract_rejects_a_solver_mode_mismatch():
    try:
        build_evaluation_contract(
            "official", "pure_cem", OFFICIAL_SOLVER_TARGETS["actor_cem"]
        )
    except ValueError as error:
        assert "requires solver target" in str(error)
    else:
        raise AssertionError("A mismatched solver target was accepted")
