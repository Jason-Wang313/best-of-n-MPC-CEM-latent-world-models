import numpy as np

from cem_refit_audit.planners import CEMPlanner, PlannerConfig
from cem_refit_audit.scoring import ScoreBatch


def score_prefers_negative(actions):
    flat_first = actions[:, 0, 0]
    score = -flat_first
    return ScoreBatch(
        score=score,
        model_return=score.copy(),
        disagreement=np.zeros_like(score),
        ood=np.zeros_like(score),
        pocket_occupancy=np.zeros_like(score),
        action_energy=np.mean(actions[:, :, 0] ** 2, axis=1),
        calibration_penalty=np.zeros_like(score),
    )


def utility_prefers_positive(actions):
    return actions[:, 0, 0]


def test_cem_is_deterministic_for_seed():
    cfg = PlannerConfig(horizon=4, population=64, iterations=3, seed=7, init_std=0.5)
    a = CEMPlanner(cfg).plan(score_prefers_negative, utility_fn=utility_prefers_positive)
    b = CEMPlanner(cfg).plan(score_prefers_negative, utility_fn=utility_prefers_positive)
    np.testing.assert_allclose(a.actions, b.actions)
    assert a.selected_score == b.selected_score


def test_cem_update_invariants():
    cfg = PlannerConfig(horizon=5, population=80, iterations=4, seed=3, min_std=0.05)
    result = CEMPlanner(cfg).plan(score_prefers_negative, utility_fn=utility_prefers_positive)
    assert len(result.history) == cfg.iterations
    assert result.samples_evaluated == cfg.population * cfg.iterations
    assert np.all(result.actions <= cfg.action_high)
    assert np.all(result.actions >= cfg.action_low)
    for trace in result.history:
        assert trace.proposal_std_mean >= cfg.min_std - 1e-12
        assert np.isfinite(trace.score_max)


def test_score_utility_separation():
    cfg = PlannerConfig(horizon=3, population=72, iterations=2, seed=11, init_std=0.8)
    result = CEMPlanner(cfg).plan(score_prefers_negative, utility_fn=utility_prefers_positive)
    assert result.selected_score > 0.0
    assert result.selected_true_return is not None
    assert result.selected_true_return < 0.0
