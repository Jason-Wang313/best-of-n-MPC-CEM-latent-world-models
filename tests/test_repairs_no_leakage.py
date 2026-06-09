import numpy as np

from boncem.planners import CEMPlanner, PlannerConfig
from boncem.scoring import ModelScorer, PilotCalibrator, RepairConfig
from boncem.worlds import ToyWorld


class GuardedToyWorld(ToyWorld):
    def true_returns(self, actions, x0=None):
        raise AssertionError("repair planning must not query evaluation labels")


def test_repaired_scorer_does_not_call_true_returns_during_planning():
    world = GuardedToyWorld()
    repair = RepairConfig(
        uncertainty_penalty=0.8,
        shadow_realism_penalty=0.4,
        disagreement_veto_quantile=0.9,
        conservative_temperature=1.3,
        elite_diversity_floor=0.2,
    )
    scorer = ModelScorer(world, repair=repair)
    cfg = PlannerConfig(horizon=world.horizon, population=48, iterations=2, seed=2)
    result = CEMPlanner(cfg, repair=repair).plan(scorer)
    assert result.selected_true_return is None


def test_calibrator_uses_only_explicit_pilot_labels():
    source_world = ToyWorld()
    guarded_world = GuardedToyWorld()
    rng = np.random.default_rng(4)
    pilot = rng.uniform(-1.0, 1.0, size=(24, source_world.horizon, source_world.action_dim))
    components = source_world.score_components(pilot)
    pilot_true = source_world.true_returns(pilot)
    calibrator = PilotCalibrator().fit(components, pilot_true)
    scorer = ModelScorer(guarded_world, repair=RepairConfig(uncertainty_penalty=0.2), calibrator=calibrator)
    cfg = PlannerConfig(horizon=guarded_world.horizon, population=48, iterations=2, seed=5)
    result = CEMPlanner(cfg, repair=RepairConfig(uncertainty_penalty=0.2)).plan(scorer)
    assert np.isfinite(result.selected_score)
