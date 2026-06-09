from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RepairConfig:
    """Model-only score corrections for CEM/Best-of-N planning."""

    uncertainty_penalty: float = 0.0
    shadow_realism_penalty: float = 0.0
    disagreement_veto_quantile: float | None = None
    conservative_temperature: float = 1.0
    elite_diversity_floor: float = 0.0
    name: str = "none"


@dataclass
class ScoreBatch:
    score: Array
    model_return: Array
    disagreement: Array
    ood: Array
    pocket_occupancy: Array
    action_energy: Array
    calibration_penalty: Array

    def as_dict(self) -> dict[str, Array]:
        return {
            "score": self.score,
            "model_return": self.model_return,
            "disagreement": self.disagreement,
            "ood": self.ood,
            "pocket_occupancy": self.pocket_occupancy,
            "action_energy": self.action_energy,
            "calibration_penalty": self.calibration_penalty,
        }


class PilotCalibrator:
    """Fits a small optimism correction from explicitly supplied pilot labels.

    The calibrator never queries an environment. Callers must provide model
    components and true returns from a separate pilot set, making label use
    auditable in tests and scripts.
    """

    def __init__(self, ridge: float = 1e-3):
        self.ridge = float(ridge)
        self.weights: Array | None = None

    @staticmethod
    def _feature_matrix(components: dict[str, Array]) -> Array:
        model_return = np.asarray(components["model_return"], dtype=float)
        cols = [
            np.ones_like(model_return),
            np.asarray(components.get("disagreement", np.zeros_like(model_return)), dtype=float),
            np.asarray(components.get("ood", np.zeros_like(model_return)), dtype=float),
            np.asarray(components.get("action_energy", np.zeros_like(model_return)), dtype=float),
        ]
        return np.stack(cols, axis=1)

    def fit(self, components: dict[str, Array], true_returns: Array) -> "PilotCalibrator":
        model_return = np.asarray(components["model_return"], dtype=float)
        true_returns = np.asarray(true_returns, dtype=float)
        if model_return.shape != true_returns.shape:
            raise ValueError("model_return and true_returns must have matching shapes")
        optimism = model_return - true_returns
        x = self._feature_matrix(components)
        reg = self.ridge * np.eye(x.shape[1])
        self.weights = np.linalg.solve(x.T @ x + reg, x.T @ optimism)
        return self

    def predict_penalty(self, components: dict[str, Array]) -> Array:
        if self.weights is None:
            raise RuntimeError("PilotCalibrator must be fit before use")
        penalty = self._feature_matrix(components) @ self.weights
        return np.maximum(0.0, penalty)


class ModelScorer:
    """Converts model components into an optimization score."""

    def __init__(
        self,
        model,
        repair: RepairConfig | None = None,
        calibrator: PilotCalibrator | None = None,
    ):
        self.model = model
        self.repair = repair or RepairConfig()
        self.calibrator = calibrator

    def __call__(self, actions: Array, x0: float | None = None) -> ScoreBatch:
        components = self.model.score_components(actions, x0)
        model_return = np.asarray(components["model_return"], dtype=float)
        disagreement = np.asarray(components.get("disagreement", np.zeros_like(model_return)), dtype=float)
        ood = np.asarray(components.get("ood", np.zeros_like(model_return)), dtype=float)
        pocket = np.asarray(components.get("pocket_occupancy", np.zeros_like(model_return)), dtype=float)
        action_energy = np.asarray(components.get("action_energy", np.zeros_like(model_return)), dtype=float)
        calibration_penalty = np.zeros_like(model_return)
        if self.calibrator is not None:
            calibration_penalty = self.calibrator.predict_penalty(components)

        score = (
            model_return
            - self.repair.uncertainty_penalty * disagreement
            - self.repair.shadow_realism_penalty * ood
            - calibration_penalty
        )
        if self.repair.disagreement_veto_quantile is not None:
            q = float(self.repair.disagreement_veto_quantile)
            if not 0.0 < q < 1.0:
                raise ValueError("disagreement_veto_quantile must be between 0 and 1")
            threshold = np.quantile(disagreement, q)
            score = np.where(disagreement > threshold, -np.inf, score)

        return ScoreBatch(
            score=np.asarray(score, dtype=float),
            model_return=model_return,
            disagreement=disagreement,
            ood=ood,
            pocket_occupancy=pocket,
            action_energy=action_energy,
            calibration_penalty=calibration_penalty,
        )
