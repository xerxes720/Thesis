# environment/learner_model.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- QualityTreeBank import (works both inside the package and as a standalone file) ---
try:
    # expected project path
    from education_framework.models.quality_tree_bank import QualityTreeBank  # type: ignore
except Exception:
    try:
        # fallback when running the file from a flat directory
        from quality_tree_bank import QualityTreeBank  # type: ignore
    except Exception:
        QualityTreeBank = None  # type: ignore


def _clip01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


# Resolve expected repo paths:
#   education_framework/
#     environment/learner_model.py   (this file)
#     models/quality_trees_assistments.joblib
BASE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE / "models" / "quality_trees_assistments.joblib"


# Dataset-aligned actions used by the decision-tree bank.
SUPPORTED_BASE_ACTIONS: Tuple[str, ...] = ("quiz", "hint", "worked_example")


DEFAULT_TUTOR_ACTION_MAP: Dict[str, str] = {
    # already aligned
    "hint": "hint",
    "worked_example": "worked_example",
    # proxies (no additional transition magnitudes are introduced here)
    "reflection_question": "quiz",
    "no_help": "quiz",
    # optional: if callers already pass a base action
    "quiz": "quiz",
}

DEFAULT_TUTEE_ACTION_MAP: Dict[str, str] = {
    # tutee requests are proxied to dataset-aligned actions
    "ask_explanation": "quiz",
    "ask_worked_example": "worked_example",
    "ask_summary": "quiz",
    "show_mistake_and_ask_fix": "quiz",
    # optional: if callers already pass a base action
    "quiz": "quiz",
    "hint": "hint",
    "worked_example": "worked_example",
}


@dataclass
class LearnerEnvConfig:
    """Environment configuration.

    Design intent:
      - Transitions are fully determined by the learned QualityTreeBank deltas.
      - Any numeric values here are *case study controls* (episode length, initialization),
        not hand-tuned transition magnitudes.
    """

    # --- case study controls ---
    max_steps: int = 500

    # Optional early termination criterion (kept out of the transition core).
    # If None, episodes end only by max_steps.
    mastery_done_threshold: Optional[float] = None

    # --- initial state (case study) ---
    init_mastery: float = 0.5
    init_rt_good: float = 0.5
    init_hint_rate: float = 0.0
    init_attempt: float = 0.0
    init_global_mastery: float = 0.5

    # Used only for logging/analysis (not included in observation by default).
    init_tutee_mastery: float = 0.0

    # --- tree bank ---
    use_quality_bank: bool = True
    model_path: Optional[str] = None

    # --- action mapping ---
    tutor_action_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TUTOR_ACTION_MAP))
    tutee_action_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TUTEE_ACTION_MAP))

    # --- reward definition (constant-free) ---
    #   - "topic_mastery_delta": reward = Δ mastery_ema[topic]
    #   - "global_mastery_delta": reward = Δ global_mastery_ema
    reward_on: str = "topic_mastery_delta"


@dataclass
class LearnerTuteeState:
    """Compact state built from ASSISTments-derived EMA features.

    Observation vector (returned to agents):
      [mastery_ema[T], rt_good_ema[T], hint_rate_ema[T], attempt_ema[T], global_mastery_ema]

    Notes:
      - This state is intentionally aligned with the decision-tree feature construction.
      - `tutee_mastery` is tracked for reporting but is not part of the observation unless
        you explicitly extend `get_observation()`.
    """

    num_topics: int

    mastery_ema: np.ndarray = field(init=False)       # [T]
    rt_good_ema: np.ndarray = field(init=False)       # [T]
    hint_rate_ema: np.ndarray = field(init=False)     # [T]
    attempt_ema: np.ndarray = field(init=False)       # [T]
    global_mastery_ema: float = 0.5

    # for reporting only
    tutee_mastery: np.ndarray = field(init=False)     # [T]

    step_count: int = 0
    assist_count: int = 0

    def __post_init__(self) -> None:
        nt = int(self.num_topics)
        # Allocate arrays; values are set in initialize_from_config().
        self.mastery_ema = np.empty(nt, dtype=np.float32)
        self.rt_good_ema = np.empty(nt, dtype=np.float32)
        self.hint_rate_ema = np.empty(nt, dtype=np.float32)
        self.attempt_ema = np.empty(nt, dtype=np.float32)
        self.tutee_mastery = np.empty(nt, dtype=np.float32)

    def initialize_from_config(self, cfg: LearnerEnvConfig) -> None:
        nt = int(self.num_topics)
        self.mastery_ema[:] = np.float32(cfg.init_mastery)
        self.rt_good_ema[:] = np.float32(cfg.init_rt_good)
        self.hint_rate_ema[:] = np.float32(cfg.init_hint_rate)
        self.attempt_ema[:] = np.float32(cfg.init_attempt)
        self.global_mastery_ema = float(cfg.init_global_mastery)
        self.tutee_mastery[:] = np.float32(cfg.init_tutee_mastery)
        self.step_count = 0
        self.assist_count = 0

    # --- compatibility helpers (to keep older training loops working) ---
    @property
    def mastery_learner(self) -> np.ndarray:
        return self.mastery_ema

    @property
    def mastery_tutee(self) -> np.ndarray:
        return self.tutee_mastery


class LearnerModel:
    """ASSISTments-aligned environment with dataset-derived transitions.

    Key properties for thesis defensibility:
      1) State representation is built from observed proxy signals (EMA features).
      2) Transition magnitudes are learned from data (QualityTreeBank deltas).
      3) Reward is constant-free: it is a direct improvement signal (Δ mastery proxy).
      4) Non-dataset actions are handled via an explicit proxy mapping *without* inventing
         new numeric magnitudes.

    Parameters
    ----------
    num_topics:
        Curriculum size.
    prereqs:
        Optional prerequisite graph (kept for future work). Currently not enforced
        in transitions to avoid introducing additional numeric assumptions.
    seed:
        Reserved (the current transition model is deterministic given the tree).
    cfg:
        Case-study controls and mappings.
    """

    def __init__(
        self,
        num_topics: int,
        prereqs: Optional[Dict[int, List[int]]] = None,
        seed: Optional[int] = 123,
        cfg: Optional[LearnerEnvConfig] = None,
    ):
        self.num_topics = int(num_topics)
        self.prereqs = prereqs or {}
        self.seed = seed
        self.cfg = cfg or LearnerEnvConfig()

        self.max_steps = int(self.cfg.max_steps)

        self.quality_bank = None
        if self.cfg.use_quality_bank:
            if QualityTreeBank is None:
                self.quality_bank = None
            else:
                model_path = self.cfg.model_path or str(DEFAULT_MODEL_PATH)
                try:
                    self.quality_bank = QualityTreeBank.load(model_path)
                except Exception:
                    # Make the environment still runnable even if the model is missing.
                    self.quality_bank = None

        self.state = LearnerTuteeState(num_topics=self.num_topics)
        self.state.initialize_from_config(self.cfg)

    # --------------- core API ----------------

    def reset(self) -> List[float]:
        self.state = LearnerTuteeState(num_topics=self.num_topics)
        self.state.initialize_from_config(self.cfg)
        return self.get_observation()

    def get_observation(self) -> List[float]:
        # Observation used by agents: 4*T + 1
        arr = np.concatenate(
            [
                self.state.mastery_ema,
                self.state.rt_good_ema,
                self.state.hint_rate_ema,
                self.state.attempt_ema,
                np.array([self.state.global_mastery_ema], dtype=np.float32),
            ]
        )
        return arr.astype(np.float32).tolist()

    def step_tutor(self, topic_id: int, action: str) -> Tuple[List[float], float, bool, Dict]:
        topic_id = int(topic_id)
        prev_topic_mastery = float(self.state.mastery_ema[topic_id])
        prev_global = float(self.state.global_mastery_ema)

        base_action = self._to_base_action(mode="tutor", action=action)
        dx = self._apply_base_action(topic_id, base_action)

        self.state.step_count += 1
        if base_action != "quiz":
            self.state.assist_count += 1

        reward = self._reward(prev_topic_mastery, prev_global, topic_id)
        done = self._check_done()

        info = {
            "mode": "tutor",
            "action": action,
            "base_action": base_action,
            "dx": dx.tolist(),
            "bank_loaded": self.quality_bank is not None,
        }
        return self.get_observation(), reward, done, info

    def step_tutee(self, topic_id: int, tutee_action: str) -> Tuple[List[float], float, bool, Dict]:
        topic_id = int(topic_id)
        prev_topic_mastery = float(self.state.mastery_ema[topic_id])
        prev_global = float(self.state.global_mastery_ema)

        base_action = self._to_base_action(mode="tutee", action=tutee_action)
        dx = self._apply_base_action(topic_id, base_action)

        # Track a simple, non-parametric tutee mastery proxy for reporting.
        # This does NOT affect the observation or reward by default.
        self.state.tutee_mastery[topic_id] = np.float32(
            _clip01(float(self.state.tutee_mastery[topic_id]) + float(dx[0]))
        )

        self.state.step_count += 1
        # A tutee interaction is still an "intervention" in the session; count it.
        self.state.assist_count += 1

        reward = self._reward(prev_topic_mastery, prev_global, topic_id)
        done = self._check_done()

        info = {
            "mode": "tutee",
            "action": tutee_action,
            "base_action": base_action,
            "dx": dx.tolist(),
            "bank_loaded": self.quality_bank is not None,
        }
        return self.get_observation(), reward, done, info

    # --------------- internal helpers ----------------

    def _to_base_action(self, mode: str, action: str) -> str:
        """Map a model action to a dataset-aligned base action.

        This function is the explicit and auditable bridge between your agent action
        vocabulary and the decision-tree bank.
        """
        if action in SUPPORTED_BASE_ACTIONS:
            return action

        if mode == "tutor":
            mapped = self.cfg.tutor_action_map.get(action)
        elif mode == "tutee":
            mapped = self.cfg.tutee_action_map.get(action)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if mapped is None:
            raise ValueError(
                f"Unknown action '{action}' for mode='{mode}'. "
                f"Either add it to the appropriate action_map or use one of {SUPPORTED_BASE_ACTIONS}."
            )

        if mapped not in SUPPORTED_BASE_ACTIONS:
            raise ValueError(
                f"Action mapping must map to one of {SUPPORTED_BASE_ACTIONS}, got '{mapped}'."
            )
        return mapped

    def _apply_base_action(self, topic_id: int, base_action: str) -> np.ndarray:
        """Apply a dataset-aligned transition.

        Features are aligned with the tree-bank training pipeline:
          x = [mastery_ema, rt_good_ema, hint_rate_ema, attempt_ema, global_mastery_ema]
        """
        if self.quality_bank is None:
            dx = np.zeros(5, dtype=np.float32)
        else:
            x = np.array(
                [
                    float(self.state.mastery_ema[topic_id]),
                    float(self.state.rt_good_ema[topic_id]),
                    float(self.state.hint_rate_ema[topic_id]),
                    float(self.state.attempt_ema[topic_id]),
                    float(self.state.global_mastery_ema),
                ],
                dtype=np.float32,
            )
            dx = self.quality_bank.predict_delta(topic_id, base_action, x).astype(np.float32)

        # Apply transition (component-wise clip to [0,1])
        self.state.mastery_ema[topic_id] = np.float32(
            _clip01(float(self.state.mastery_ema[topic_id]) + float(dx[0]))
        )
        self.state.rt_good_ema[topic_id] = np.float32(
            _clip01(float(self.state.rt_good_ema[topic_id]) + float(dx[1]))
        )
        self.state.hint_rate_ema[topic_id] = np.float32(
            _clip01(float(self.state.hint_rate_ema[topic_id]) + float(dx[2]))
        )
        self.state.attempt_ema[topic_id] = np.float32(
            _clip01(float(self.state.attempt_ema[topic_id]) + float(dx[3]))
        )
        self.state.global_mastery_ema = _clip01(float(self.state.global_mastery_ema) + float(dx[4]))

        return dx

    def _reward(self, prev_topic_mastery: float, prev_global: float, topic_id: int) -> float:
        """Constant-free reward based on state improvement."""
        if self.cfg.reward_on == "global_mastery_delta":
            return float(self.state.global_mastery_ema - prev_global)

        # default: topic mastery delta
        return float(self.state.mastery_ema[topic_id] - prev_topic_mastery)

    def _check_done(self) -> bool:
        if self.state.step_count >= self.max_steps:
            return True
        thr = self.cfg.mastery_done_threshold
        if thr is None:
            return False
        # Early completion criterion (case-study optional)
        return bool(np.all(self.state.mastery_ema >= np.float32(thr)))
