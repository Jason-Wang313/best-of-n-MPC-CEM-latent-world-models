from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from boncem.scoring import RepairConfig, ScoreBatch


Array = np.ndarray
ScoreFn = Callable[[Array], ScoreBatch]
UtilityFn = Callable[[Array], Array]


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 8
    action_dim: int = 1
    action_low: float = -1.0
    action_high: float = 1.0
    population: int = 256
    elite_fraction: float = 0.1
    iterations: int = 5
    seed: int = 0
    init_std: float = 0.85
    min_std: float = 0.04
    momentum: float = 0.0

    @property
    def elite_count(self) -> int:
        return max(1, int(np.ceil(self.population * self.elite_fraction)))


@dataclass
class IterationTrace:
    iteration: int
    score_max: float
    score_mean: float
    selected_model_return: float
    selected_true_return: float | None
    selected_tail_gap: float | None
    elite_score_mean: float
    elite_true_mean: float | None
    elite_model_error_mean: float | None
    elite_pocket_mean: float
    elite_disagreement_mean: float
    elite_diversity: float
    proposal_drift: float
    proposal_std_mean: float


@dataclass
class PlanResult:
    planner: str
    actions: Array
    selected_score: float
    selected_model_return: float
    selected_true_return: float | None
    samples_evaluated: int
    history: list[IterationTrace] = field(default_factory=list)
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    def to_row(self) -> dict[str, float | int | str | None]:
        final = self.history[-1] if self.history else None
        row: dict[str, float | int | str | None] = {
            "planner": self.planner,
            "selected_score": self.selected_score,
            "selected_model_return": self.selected_model_return,
            "selected_true_return": self.selected_true_return,
            "samples_evaluated": self.samples_evaluated,
            "first_action": float(self.actions[0, 0]),
        }
        if final is not None:
            row.update(
                {
                    "selected_tail_gap": final.selected_tail_gap,
                    "elite_model_error_mean": final.elite_model_error_mean,
                    "elite_pocket_mean": final.elite_pocket_mean,
                    "elite_disagreement_mean": final.elite_disagreement_mean,
                    "elite_diversity": final.elite_diversity,
                    "proposal_drift": final.proposal_drift,
                    "proposal_std_mean": final.proposal_std_mean,
                }
            )
        row.update(self.metadata)
        return row


def _finite_argmax(scores: Array) -> int:
    if np.all(~np.isfinite(scores)):
        raise RuntimeError("all candidate scores were non-finite")
    return int(np.nanargmax(scores))


def _elite_indices(scores: Array, elite_count: int, samples: Array, diversity_floor: float = 0.0) -> Array:
    order = np.argsort(scores)[::-1]
    order = order[np.isfinite(scores[order])]
    if len(order) == 0:
        raise RuntimeError("cannot select elites from all non-finite scores")
    if diversity_floor <= 0.0:
        return order[:elite_count]

    flat = samples.reshape(samples.shape[0], -1)
    chosen: list[int] = []
    for idx in order:
        if not chosen:
            chosen.append(int(idx))
        else:
            distances = np.linalg.norm(flat[idx] - flat[np.asarray(chosen)], axis=1)
            if float(np.min(distances)) >= diversity_floor:
                chosen.append(int(idx))
        if len(chosen) >= elite_count:
            break
    if len(chosen) < elite_count:
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= elite_count:
                break
    return np.asarray(chosen[:elite_count], dtype=int)


def _trace(
    iteration: int,
    samples: Array,
    scores: ScoreBatch,
    selected_idx: int,
    elite_idx: Array,
    mean: Array,
    std: Array,
    initial_mean: Array,
    utility_fn: UtilityFn | None,
) -> IterationTrace:
    selected_true: float | None = None
    elite_true_mean: float | None = None
    elite_model_error_mean: float | None = None
    selected_tail_gap: float | None = None
    if utility_fn is not None:
        selected_true_values = utility_fn(samples[[selected_idx]])
        selected_true = float(selected_true_values[0])
        elite_true = utility_fn(samples[elite_idx])
        elite_true_mean = float(np.mean(elite_true))
        elite_model_error_mean = float(np.mean(scores.model_return[elite_idx] - elite_true))
        selected_tail_gap = float(scores.model_return[selected_idx] - selected_true)
    return IterationTrace(
        iteration=iteration,
        score_max=float(np.nanmax(scores.score)),
        score_mean=float(np.nanmean(scores.score[np.isfinite(scores.score)])),
        selected_model_return=float(scores.model_return[selected_idx]),
        selected_true_return=selected_true,
        selected_tail_gap=selected_tail_gap,
        elite_score_mean=float(np.mean(scores.score[elite_idx])),
        elite_true_mean=elite_true_mean,
        elite_model_error_mean=elite_model_error_mean,
        elite_pocket_mean=float(np.mean(scores.pocket_occupancy[elite_idx])),
        elite_disagreement_mean=float(np.mean(scores.disagreement[elite_idx])),
        elite_diversity=float(np.mean(np.std(samples[elite_idx].reshape(len(elite_idx), -1), axis=0))),
        proposal_drift=float(np.linalg.norm(mean - initial_mean)),
        proposal_std_mean=float(np.mean(std)),
    )


class RandomShootingPlanner:
    def __init__(self, config: PlannerConfig):
        self.config = config

    def plan(self, score_fn: ScoreFn, utility_fn: UtilityFn | None = None) -> PlanResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        samples = rng.uniform(
            cfg.action_low,
            cfg.action_high,
            size=(cfg.population, cfg.horizon, cfg.action_dim),
        )
        scores = score_fn(samples)
        selected_idx = _finite_argmax(scores.score)
        elites = _elite_indices(scores.score, cfg.elite_count, samples)
        mean = np.mean(samples[elites], axis=0)
        std = np.std(samples[elites], axis=0)
        trace = _trace(0, samples, scores, selected_idx, elites, mean, std, np.zeros_like(mean), utility_fn)
        selected_true = trace.selected_true_return
        return PlanResult(
            planner="random_shooting",
            actions=samples[selected_idx],
            selected_score=float(scores.score[selected_idx]),
            selected_model_return=float(scores.model_return[selected_idx]),
            selected_true_return=selected_true,
            samples_evaluated=cfg.population,
            history=[trace],
            metadata={"config": str(asdict(cfg))},
        )


class BestOfNPlanner:
    def __init__(self, config: PlannerConfig):
        self.config = config

    def plan(self, score_fn: ScoreFn, utility_fn: UtilityFn | None = None) -> PlanResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        samples = rng.normal(
            loc=0.0,
            scale=cfg.init_std,
            size=(cfg.population, cfg.horizon, cfg.action_dim),
        )
        samples = np.clip(samples, cfg.action_low, cfg.action_high)
        scores = score_fn(samples)
        selected_idx = _finite_argmax(scores.score)
        elites = _elite_indices(scores.score, cfg.elite_count, samples)
        mean = np.mean(samples[elites], axis=0)
        std = np.std(samples[elites], axis=0)
        trace = _trace(0, samples, scores, selected_idx, elites, mean, std, np.zeros_like(mean), utility_fn)
        return PlanResult(
            planner="best_of_n",
            actions=samples[selected_idx],
            selected_score=float(scores.score[selected_idx]),
            selected_model_return=float(scores.model_return[selected_idx]),
            selected_true_return=trace.selected_true_return,
            samples_evaluated=cfg.population,
            history=[trace],
            metadata={"config": str(asdict(cfg))},
        )


class CEMPlanner:
    def __init__(self, config: PlannerConfig, repair: RepairConfig | None = None, name: str = "cem"):
        self.config = config
        self.repair = repair or RepairConfig()
        self.name = name

    def plan(self, score_fn: ScoreFn, utility_fn: UtilityFn | None = None) -> PlanResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        mean = np.zeros((cfg.horizon, cfg.action_dim), dtype=float)
        initial_mean = mean.copy()
        std = np.full_like(mean, cfg.init_std)
        history: list[IterationTrace] = []
        best_sample: Array | None = None
        best_score = -np.inf
        best_model_return = -np.inf
        best_true_return: float | None = None
        samples_evaluated = 0

        for iteration in range(cfg.iterations):
            samples = rng.normal(mean, std, size=(cfg.population, cfg.horizon, cfg.action_dim))
            samples = np.clip(samples, cfg.action_low, cfg.action_high)
            scores = score_fn(samples)
            selected_idx = _finite_argmax(scores.score)
            elite_idx = _elite_indices(
                scores.score,
                cfg.elite_count,
                samples,
                diversity_floor=self.repair.elite_diversity_floor,
            )
            if float(scores.score[selected_idx]) > best_score:
                best_score = float(scores.score[selected_idx])
                best_sample = samples[selected_idx].copy()
                best_model_return = float(scores.model_return[selected_idx])
                if utility_fn is not None:
                    best_true_return = float(utility_fn(samples[[selected_idx]])[0])

            elite_samples = samples[elite_idx]
            elite_mean = np.mean(elite_samples, axis=0)
            elite_std = np.std(elite_samples, axis=0)
            temperature = max(1.0, float(self.repair.conservative_temperature))
            elite_std = elite_std * temperature
            elite_std = np.maximum(elite_std, cfg.min_std)
            momentum = float(cfg.momentum)
            mean = momentum * mean + (1.0 - momentum) * elite_mean
            std = momentum * std + (1.0 - momentum) * elite_std
            std = np.maximum(std, cfg.min_std)
            trace = _trace(
                iteration,
                samples,
                scores,
                selected_idx,
                elite_idx,
                mean,
                std,
                initial_mean,
                utility_fn,
            )
            history.append(trace)
            samples_evaluated += cfg.population

        if best_sample is None:
            raise RuntimeError("CEM did not evaluate any finite samples")
        return PlanResult(
            planner=self.name,
            actions=best_sample,
            selected_score=best_score,
            selected_model_return=best_model_return,
            selected_true_return=best_true_return,
            samples_evaluated=samples_evaluated,
            history=history,
            metadata={
                "repair": self.repair.name,
                "config": str(asdict(cfg)),
            },
        )
