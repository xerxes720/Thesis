# learner_model.py
"""
KDD Algebra learner simulator (case-study implementation)

Design goals
------------
1) Dataset-native state:
   - Built only from signals available in KDD/Bridge-to-Algebra style logs:
     CFA (correct first attempt), hints/incorrects/corrects counts, step duration, KC tags, opportunity counts.

2) Decision-tree-compatible:
   - Provides a feature schema and trajectory builder that can be used by a build_decision_tree script
     to train (a) response model(s) and (b) transition/update model(s).
   - Trees can be constrained to max_depth=7 to mirror the paper's simulator complexity.

3) No arbitrary "protégé bonus constant":
   - Tutee effects are implemented through a "generative interaction" mechanism (generation_mode)
     that can be estimated from data (via quantiles / propensity model) and passed as a feature into the
     transition model. The simulator then uses the learned transition model, not a hand-picked additive term.

What this module provides
-------------------------
- KDDActionSchema: data-driven thresholds for action labeling and for "generation_mode" proxies.
- LearnerState: per-topic state arrays (mastery, EMAs, opportunity).
- KDDLearnerModel: a simulator that steps using learned per-topic models:
    * response_model(topic): predicts probability(CFA=1)
    * transition_model(topic): predicts delta-state vector given features and observed outcome
    * optional auxiliary models for hint/time/incorrect counts if you train them (not required).
- KDDTrajectoryBuilder: converts KDD logs to supervised training rows for the above models.

Assumptions you can keep stable for the committee
-------------------------------------------------
- 8 macro-topics (subtasks) are derived from KCs by a fixed mapping you report (clustering or domain mapping).
- State is an estimator of recent performance and behavior (EMA-based); alpha can be cross-validated in training.
- Actions are abstract instructional "modes" aligned with observable patterns (hints/time/errors) in KDD.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import math
import random

import numpy as np

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


# ----------------------------
# Utilities
# ----------------------------

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


# ----------------------------
# Action space (|A_l| = 8)
# ----------------------------

class LowLevelAction(IntEnum):
    """
    8 low-level instructional modes.
    Keep this fixed if you want paper-like comparability (|A_l| = 8).

    You may later decide which two are "tutee" actions for the added contribution;
    the simulator supports marking an action as "tutee-like" via ActionMeta (see below).
    """
    INDEPENDENT_PRACTICE = 0
    SCAFFOLDED_PRACTICE = 1
    WORKED_EXAMPLE_THEN_PRACTICE = 2
    ERROR_FOCUSED_REMEDIATION = 3
    SPACED_REVIEW = 4
    FLUENCY_DRILL = 5
    CHALLENGE_PROBLEM = 6
    TEACH_BACK_OR_DIAGNOSE = 7  # reserved for your tutee contribution (generative interaction)


@dataclass(frozen=True)
class ActionMeta:
    """
    Meta attributes for an action. These do NOT add any constant bonus.
    They only provide context features that your transition model can learn from.

    - is_tutee: marks whether the interaction is with a tutee (social teaching context).
    - force_generation: if True, the interaction is definitionally generative (teach-back/diagnose).
      This affects the generation_mode feature (see KDDActionSchema.generation_mode()).
    """
    action: LowLevelAction
    is_tutee: bool = False
    force_generation: bool = False


# ----------------------------
# Dataset-driven labeling schema
# ----------------------------

@dataclass
class KDDActionSchema:
    """
    Data-driven thresholds for:
    - labeling KDD rows into abstract action modes (for training the simulator models)
    - defining a "generation_mode" proxy (for defendable tutee integration)

    Fit this schema ONCE from the training split of KDD logs.
    Store the quantiles and reuse for labeling and simulation.
    """
    # duration quantiles in seconds
    dur_q25: float = 5.0
    dur_q50: float = 15.0
    dur_q75: float = 35.0
    dur_q90: float = 60.0

    # hints quantiles
    hints_q50: float = 0.0
    hints_q75: float = 1.0

    # incorrects quantiles
    inc_q50: float = 0.0
    inc_q75: float = 1.0

    # corrects quantiles (often 0/1; still keep for completeness)
    cor_q50: float = 1.0

    @staticmethod
    def _quantile(values: Sequence[float], q: float, default: float) -> float:
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
        if arr.size == 0:
            return default
        return float(np.quantile(arr, q))

    @classmethod
    def fit_from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        duration_col: str = "Step Duration (sec)",
        hints_col: str = "Hints",
        incorrects_col: str = "Incorrects",
        corrects_col: str = "Corrects",
        max_rows: int = 2_000_000,
    ) -> "KDDActionSchema":
        durs: List[float] = []
        hints: List[float] = []
        incs: List[float] = []
        cors: List[float] = []

        for i, r in enumerate(rows):
            if i >= max_rows:
                break
            durs.append(_safe_float(r.get(duration_col), 0.0))
            hints.append(_safe_float(r.get(hints_col), 0.0))
            incs.append(_safe_float(r.get(incorrects_col), 0.0))
            cors.append(_safe_float(r.get(corrects_col), 0.0))

        schema = cls()
        schema.dur_q25 = cls._quantile(durs, 0.25, schema.dur_q25)
        schema.dur_q50 = cls._quantile(durs, 0.50, schema.dur_q50)
        schema.dur_q75 = cls._quantile(durs, 0.75, schema.dur_q75)
        schema.dur_q90 = cls._quantile(durs, 0.90, schema.dur_q90)

        schema.hints_q50 = cls._quantile(hints, 0.50, schema.hints_q50)
        schema.hints_q75 = cls._quantile(hints, 0.75, schema.hints_q75)

        schema.inc_q50 = cls._quantile(incs, 0.50, schema.inc_q50)
        schema.inc_q75 = cls._quantile(incs, 0.75, schema.inc_q75)

        schema.cor_q50 = cls._quantile(cors, 0.50, schema.cor_q50)
        return schema

    def label_action_from_row(
        self,
        row: Mapping[str, Any],
        cfa_col: str = "Correct First Attempt",
        duration_col: str = "Step Duration (sec)",
        hints_col: str = "Hints",
        incorrects_col: str = "Incorrects",
    ) -> LowLevelAction:
        """
        Rule-based labeling that is:
        - transparent
        - uses only dataset-driven thresholds (quantiles)
        - yields one of 8 abstract actions for training

        You can later refine this mapping; keep it fixed for reproduction runs.
        """
        cfa = _safe_int(row.get(cfa_col), 0)
        dur = _safe_float(row.get(duration_col), 0.0)
        h = _safe_int(row.get(hints_col), 0)
        inc = _safe_int(row.get(incorrects_col), 0)

        # Strong struggle signal -> remediation
        if inc > self.inc_q75:
            return LowLevelAction.ERROR_FOCUSED_REMEDIATION

        # Hint usage -> scaffolded practice
        if h > self.hints_q50:
            return LowLevelAction.SCAFFOLDED_PRACTICE

        # Very short steps -> fluency drill (speed/automaticity)
        if dur > 0 and dur <= self.dur_q25 and cfa == 1:
            return LowLevelAction.FLUENCY_DRILL

        # Very long steps (effortful) can be either challenge or worked-example; use CFA to separate
        if dur >= self.dur_q75:
            return LowLevelAction.CHALLENGE_PROBLEM if cfa == 1 else LowLevelAction.WORKED_EXAMPLE_THEN_PRACTICE

        # Default: independent practice
        return LowLevelAction.INDEPENDENT_PRACTICE

    def generation_mode_from_row(
        self,
        row: Mapping[str, Any],
        cfa_col: str = "Correct First Attempt",
        duration_col: str = "Step Duration (sec)",
        hints_col: str = "Hints",
        incorrects_col: str = "Incorrects",
    ) -> int:
        """
        A defensible proxy for "generative / explanation-like" engagement derived from KDD signals.

        Logic:
        - Must be correct on first attempt (CFA=1)
        - Must be low-hint (hints <= median)
        - Must not be instant-guessing (duration >= q25)
        - Must not be extremely slow / off-task (duration <= q90)
        - Must not have many incorrects (incorrects <= median)

        This yields a binary feature that your transition model can learn from.
        For tutee actions, you can force generation_mode=1 (definitionally teach-back).
        """
        cfa = _safe_int(row.get(cfa_col), 0)
        dur = _safe_float(row.get(duration_col), 0.0)
        h = _safe_int(row.get(hints_col), 0)
        inc = _safe_int(row.get(incorrects_col), 0)

        if cfa != 1:
            return 0
        if h > self.hints_q50:
            return 0
        if dur < self.dur_q25 or dur > self.dur_q90:
            return 0
        if inc > self.inc_q50:
            return 0
        return 1


# ----------------------------
# Learner state
# ----------------------------

@dataclass
class LearnerState:
    """
    Per-topic state for 8 subtasks.

    mastery[k] is an *estimated probability of CFA correctness* for topic k (0..1).
    The rest are behavior/fluency proxies derived from KDD signals.
    """
    n_topics: int = 8

    mastery: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    opp: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.int32))  # opportunity count per topic
    cfa_ema: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    hint_ema: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    time_ema: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    inc_ema: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))

    total_steps: int = 0

    def copy(self) -> "LearnerState":
        return LearnerState(
            n_topics=self.n_topics,
            mastery=self.mastery.copy(),
            opp=self.opp.copy(),
            cfa_ema=self.cfa_ema.copy(),
            hint_ema=self.hint_ema.copy(),
            time_ema=self.time_ema.copy(),
            inc_ema=self.inc_ema.copy(),
            total_steps=int(self.total_steps),
        )


# ----------------------------
# Model bundle (trained elsewhere)
# ----------------------------

@dataclass
class KDDModelBundle:
    """
    Container you will save after build_decision_tree training.

    Expected sklearn-like interface:
    - response_models[k].predict_proba(X) -> [:, 1] for P(CFA=1)
    - transition_models[k].predict(X) -> delta vector per sample (float array)

    Optional:
    - aux models for hints/time/incorrects if you train them.
    """
    n_topics: int
    schema: KDDActionSchema

    # maps raw KC string -> topic id 0..n_topics-1
    kc_to_topic: Dict[str, int]

    response_models: Dict[int, Any]
    transition_models: Dict[int, Any]

    # Optional auxiliary outcome models (predict expected hints/time/inc given X)
    hints_models: Optional[Dict[int, Any]] = None
    time_models: Optional[Dict[int, Any]] = None
    inc_models: Optional[Dict[int, Any]] = None

    # Feature options used in training; store so inference matches training
    one_hot_actions: bool = True

    # EMA smoothing used to define state from history; ideally chosen via CV in training
    ema_alpha: float = 0.2


# ----------------------------
# Learner simulator
# ----------------------------

@dataclass
class KDDLearnerConfig:
    n_topics: int = 8

    # Episode completion criteria (for your environment / training loop)
    mastery_threshold: float = 0.65
    opp_min: int = 3

    # Reward shaping knobs (you can set these in your environment wrapper;
    # included here for convenience if you simulate reward at this layer)
    step_penalty: float = -0.01
    correct_reward: float = 0.02
    completion_reward: float = 3.0

    # Normalization constants (derived from dataset or set conservatively)
    opp_norm: float = 20.0
    time_norm: float = 120.0
    hints_norm: float = 5.0
    inc_norm: float = 5.0


class KDDLearnerModel:
    """
    KDD-based learner simulator.

    This model is designed to be controlled by your hierarchical agents:
    - HL chooses topic k in {0..7}
    - LL chooses LowLevelAction in {0..7} (wrapped by ActionMeta)
    """

    def __init__(
        self,
        cfg: Optional[KDDLearnerConfig] = None,
        bundle: Optional[KDDModelBundle] = None,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or KDDLearnerConfig()
        self.bundle = bundle
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.state = LearnerState(n_topics=self.cfg.n_topics)

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        if joblib is None:
            raise RuntimeError("joblib not available; install joblib to save model bundles.")
        joblib.dump({"cfg": self.cfg, "bundle": self.bundle}, path)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> "KDDLearnerModel":
        if joblib is None:
            raise RuntimeError("joblib not available; install joblib to load model bundles.")
        obj = joblib.load(path)
        model = cls(cfg=obj["cfg"], bundle=obj["bundle"], seed=seed)
        return model

    # ---------- environment-like API ----------
    def reset(self, initial_mastery: Union[float, Sequence[float]] = 0.2) -> LearnerState:
        n = self.cfg.n_topics
        if isinstance(initial_mastery, (list, tuple, np.ndarray)):
            arr = np.asarray(initial_mastery, dtype=np.float32)
            if arr.shape[0] != n:
                raise ValueError(f"initial_mastery length must be {n}")
            self.state.mastery = np.clip(arr, 0.0, 1.0)
        else:
            self.state.mastery = np.full(n, float(initial_mastery), dtype=np.float32)

        self.state.opp = np.zeros(n, dtype=np.int32)
        self.state.cfa_ema = self.state.mastery.copy()
        self.state.hint_ema = np.zeros(n, dtype=np.float32)
        self.state.time_ema = np.zeros(n, dtype=np.float32)
        self.state.inc_ema = np.zeros(n, dtype=np.float32)
        self.state.total_steps = 0
        return self.state.copy()

    def is_done(self) -> bool:
        s = self.state
        complete = (s.mastery >= self.cfg.mastery_threshold) & (s.opp >= self.cfg.opp_min)
        return bool(np.all(complete))

    def step(
        self,
        topic_id: int,
        action_meta: ActionMeta,
    ) -> Tuple[LearnerState, Dict[str, Any]]:
        """
        Perform one simulated interaction on a topic using learned models.

        Returns:
            next_state (copy),
            info dict with:
              - p_correct, cfa, hints, incorrects, duration, generation_mode
              - reward (optional simple shaping; you can ignore if reward handled elsewhere)
              - done
        """
        if self.bundle is None:
            raise RuntimeError("KDDLearnerModel.bundle is None. Load or provide a trained KDDModelBundle first.")
        if topic_id < 0 or topic_id >= self.cfg.n_topics:
            raise ValueError(f"topic_id must be in [0, {self.cfg.n_topics-1}]")

        s = self.state
        x_base = self._build_features(
            s=s,
            topic_id=topic_id,
            action_meta=action_meta,
            generation_mode=0,  # unknown until we simulate; provided later to transition model
            include_outcome=False,
        )

        # 1) Predict P(CFA=1) and sample outcome
        resp_model = self.bundle.response_models.get(topic_id)
        if resp_model is None:
            # fallback: use mastery as probability (should not happen in final experiments)
            p_correct = float(s.mastery[topic_id])
        else:
            p_correct = self._predict_proba_1(resp_model, x_base)

        cfa = 1 if self.rng.random() < p_correct else 0

        # 2) Simulate auxiliary outcomes if models exist; else heuristics
        hints = self._predict_aux_int(topic_id, x_base, self.bundle.hints_models, default=0)
        incorrects = self._predict_aux_int(topic_id, x_base, self.bundle.inc_models, default=(0 if cfa == 1 else 1))
        duration = self._predict_aux_float(topic_id, x_base, self.bundle.time_models, default=self.bundle.schema.dur_q50)

        # 3) Determine generation_mode (data-driven proxy)
        if action_meta.force_generation:
            generation_mode = 1
        else:
            # Use an interpretable proxy: generation is more likely when correct, low hints, and duration not extreme.
            # Here we approximate using schema thresholds and simulated outcomes (still dataset-tethered).
            generation_mode = int(
                (cfa == 1)
                and (hints <= self.bundle.schema.hints_q50)
                and (duration >= self.bundle.schema.dur_q25)
                and (duration <= self.bundle.schema.dur_q90)
                and (incorrects <= self.bundle.schema.inc_q50)
            )

        # 4) Predict delta-state from transition model given outcome + generation_mode
        x_trans = self._build_features(
            s=s,
            topic_id=topic_id,
            action_meta=action_meta,
            generation_mode=generation_mode,
            include_outcome=True,
            cfa=cfa,
            hints=hints,
            incorrects=incorrects,
            duration=duration,
        )

        trans_model = self.bundle.transition_models.get(topic_id)
        if trans_model is None:
            # fallback: small mastery update based on CFA (should not happen in final experiments)
            delta = np.zeros(self._delta_dim(), dtype=np.float32)
            delta[0] = 0.02 * (1.0 - float(s.mastery[topic_id])) if cfa == 1 else -0.01 * float(s.mastery[topic_id])
        else:
            delta = np.asarray(trans_model.predict(x_trans.reshape(1, -1))[0], dtype=np.float32)
            if delta.shape[0] != self._delta_dim():
                raise RuntimeError(f"transition model returned delta dim {delta.shape[0]}, expected {self._delta_dim()}")

        # 5) Apply delta + deterministic bookkeeping updates
        self._apply_delta(topic_id, delta, cfa=cfa, hints=hints, incorrects=incorrects, duration=duration)

        # 6) Optional shaping reward at this layer (you may override elsewhere)
        done = self.is_done()
        reward = float(self.cfg.step_penalty + (self.cfg.correct_reward if cfa == 1 else 0.0) + (self.cfg.completion_reward if done else 0.0))

        info = {
            "p_correct": p_correct,
            "cfa": cfa,
            "hints": hints,
            "incorrects": incorrects,
            "duration": duration,
            "generation_mode": generation_mode,
            "reward": reward,
            "done": done,
        }
        return self.state.copy(), info

    # ---------- feature engineering ----------
    def feature_dim(self) -> int:
        # base features + optional one-hot action
        base = 10  # per-topic mastery, opp, EMAs + global mastery + total steps norm + is_tutee + generation_mode
        if self.bundle is None:
            one_hot = 8
        else:
            one_hot = 8 if self.bundle.one_hot_actions else 1
        # outcome features appended in transition input: cfa, hints, incorrects, duration_norm
        return base + one_hot + 4

    def _delta_dim(self) -> int:
        """
        Transition model output dimension.

        Keep this small and interpretable. Recommended:
        - delta_mastery
        - delta_cfa_ema
        - delta_hint_ema
        - delta_time_ema
        - delta_inc_ema

        Opportunity and total_steps are deterministic bookkeeping (not predicted).
        """
        return 5

    def _build_features(
        self,
        s: LearnerState,
        topic_id: int,
        action_meta: ActionMeta,
        generation_mode: int,
        include_outcome: bool,
        cfa: int = 0,
        hints: int = 0,
        incorrects: int = 0,
        duration: float = 0.0,
    ) -> np.ndarray:
        cfg = self.cfg

        mastery_k = float(s.mastery[topic_id])
        opp_k = float(s.opp[topic_id]) / max(cfg.opp_norm, 1e-6)
        cfa_ema_k = float(s.cfa_ema[topic_id])
        hint_ema_k = float(s.hint_ema[topic_id])
        time_ema_k = float(s.time_ema[topic_id])
        inc_ema_k = float(s.inc_ema[topic_id])

        global_mastery = float(np.mean(s.mastery))
        total_steps_norm = float(s.total_steps) / (cfg.opp_norm * cfg.n_topics)

        is_tutee = 1.0 if action_meta.is_tutee else 0.0
        gen = float(generation_mode)

        feats: List[float] = [
            mastery_k,
            opp_k,
            cfa_ema_k,
            hint_ema_k,
            time_ema_k,
            inc_ema_k,
            global_mastery,
            total_steps_norm,
            is_tutee,
            gen,
        ]

        # action encoding
        if self.bundle is None or self.bundle.one_hot_actions:
            a = int(action_meta.action)
            one_hot = [0.0] * 8
            one_hot[a] = 1.0
            feats.extend(one_hot)
        else:
            feats.append(float(int(action_meta.action)) / 7.0)

        # outcome features (only used for transition model input)
        if include_outcome:
            feats.extend([
                float(cfa),
                float(hints) / max(cfg.hints_norm, 1e-6),
                float(incorrects) / max(cfg.inc_norm, 1e-6),
                float(duration) / max(cfg.time_norm, 1e-6),
            ])
        else:
            # keep shape consistent for response model input; append zeros
            feats.extend([0.0, 0.0, 0.0, 0.0])

        return np.asarray(feats, dtype=np.float32)

    # ---------- model prediction helpers ----------
    def _predict_proba_1(self, model: Any, x: np.ndarray) -> float:
        # sklearn DecisionTreeClassifier / similar
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x.reshape(1, -1))
            # assume binary with classes [0,1]
            return float(proba[0, 1])
        # fallback: regressor producing probability
        y = float(model.predict(x.reshape(1, -1))[0])
        return _clip01(y)

    def _predict_aux_int(
        self,
        topic_id: int,
        x: np.ndarray,
        models: Optional[Dict[int, Any]],
        default: int,
    ) -> int:
        if models is None:
            return int(default)
        m = models.get(topic_id)
        if m is None:
            return int(default)
        y = float(m.predict(x.reshape(1, -1))[0])
        return max(0, int(round(y)))

    def _predict_aux_float(
        self,
        topic_id: int,
        x: np.ndarray,
        models: Optional[Dict[int, Any]],
        default: float,
    ) -> float:
        if models is None:
            return float(default)
        m = models.get(topic_id)
        if m is None:
            return float(default)
        y = float(m.predict(x.reshape(1, -1))[0])
        return max(0.0, y)

    # ---------- state update ----------
    def _apply_delta(
        self,
        topic_id: int,
        delta: np.ndarray,
        cfa: int,
        hints: int,
        incorrects: int,
        duration: float,
    ) -> None:
        """
        Apply learned delta to the per-topic latent state, then update observable EMAs from the realized outcomes.
        This separates:
        - latent learning effect (learned from KDD via transition model)
        - immediate observations (deterministic EMA updates)
        """
        b = self.bundle
        assert b is not None

        alpha = float(b.ema_alpha)

        # learned deltas
        d_mastery, d_cfa_ema, d_hint_ema, d_time_ema, d_inc_ema = [float(x) for x in delta.tolist()]

        s = self.state
        s.mastery[topic_id] = _clip01(float(s.mastery[topic_id]) + d_mastery)
        s.cfa_ema[topic_id] = _clip01(float(s.cfa_ema[topic_id]) + d_cfa_ema)
        s.hint_ema[topic_id] = _clip01(float(s.hint_ema[topic_id]) + d_hint_ema)
        s.time_ema[topic_id] = _clip01(float(s.time_ema[topic_id]) + d_time_ema)
        s.inc_ema[topic_id] = _clip01(float(s.inc_ema[topic_id]) + d_inc_ema)

        # deterministic bookkeeping updates from realized outcomes
        s.opp[topic_id] += 1
        s.total_steps += 1

        # EMA updates (observational, not "bonus")
        # These EMAs can also be used as targets/inputs in training; alpha should be selected via CV.
        # s.cfa_ema[topic_id] = _clip01((1.0 - alpha) * float(s.cfa_ema[topic_id]) + alpha * float(cfa))
        # s.hint_ema[topic_id] = _clip01((1.0 - alpha) * float(s.hint_ema[topic_id]) + alpha * (float(hints) / max(self.cfg.hints_norm, 1e-6)))
        # s.time_ema[topic_id] = _clip01((1.0 - alpha) * float(s.time_ema[topic_id]) + alpha * (float(duration) / max(self.cfg.time_norm, 1e-6)))
        # s.inc_ema[topic_id] = _clip01((1.0 - alpha) * float(s.inc_ema[topic_id]) + alpha * (float(incorrects) / max(self.cfg.inc_norm, 1e-6)))


# ----------------------------
# Trajectory builder for training trees
# ----------------------------

@dataclass
class TrainingRow:
    topic_id: int
    action_id: int
    x_resp: np.ndarray
    cfa: int
    x_trans: np.ndarray
    delta: np.ndarray
    hints: int
    incorrects: int
    duration: float



class KDDTrajectoryBuilder:
    """
    Converts KDD logs into training rows for per-topic decision trees.

    This builder:
    - Maintains a deterministic state estimator (EMA-based)
    - Labels each row with an abstract action (via KDDActionSchema)
    - Computes generation_mode proxy from dataset
    - Produces (x_resp, cfa) and (x_trans, delta) samples

    Note: The delta target is defined in terms of how the state estimator changes after incorporating the row.
          This is what makes the simulator "learned": transition trees learn those deltas from data.
    """

    def __init__(
        self,
        n_topics: int,
        kc_to_topic: Mapping[str, int],
        schema: KDDActionSchema,
        ema_alpha: float = 0.2,
        one_hot_actions: bool = True,
        seed: int = 0,
    ) -> None:
        self.n_topics = n_topics
        self.kc_to_topic = dict(kc_to_topic)
        self.schema = schema
        self.ema_alpha = float(ema_alpha)
        self.one_hot_actions = bool(one_hot_actions)
        self.rng = random.Random(seed)

        # learner model used ONLY for feature construction and deterministic estimator updates in training extraction
        self.sim = KDDLearnerModel(
            cfg=KDDLearnerConfig(n_topics=n_topics),
            bundle=KDDModelBundle(
                n_topics=n_topics,
                schema=schema,
                kc_to_topic=dict(kc_to_topic),
                response_models={},          # not needed for building training rows
                transition_models={},        # not needed for building training rows
                one_hot_actions=one_hot_actions,
                ema_alpha=ema_alpha,
            ),
            seed=seed,
        )
        self.sim.reset(initial_mastery=0.2)

    def _extract_kc(self, row: Mapping[str, Any], kc_col: str) -> Optional[str]:
        raw = row.get(kc_col)
        if raw is None:
            return None
        s = str(raw).strip()
        if s == "":
            return None
        # KDD sometimes has multiple KCs separated by "~~"; choose primary KC for simplicity
        if "~~" in s:
            s = s.split("~~")[0].strip()
        return s

    def iter_training_rows(
        self,
        rows_by_student: Iterable[Tuple[str, List[Mapping[str, Any]]]],
        *,
        kc_col: str = "KC(Default)",
        cfa_col: str = "Correct First Attempt",
        duration_col: str = "Step Duration (sec)",
        hints_col: str = "Hints",
        incorrects_col: str = "Incorrects",
        # optional: if your dataset provides explicit opportunity per KC
        opportunity_col: Optional[str] = None,

    ) -> Iterable[TrainingRow]:
        """
        rows_by_student must yield (student_id, rows_sorted_in_time).
        Sorting should be done by your loader using columns like (row index, timestamp, problem view).
        """

        for _sid, seq in rows_by_student:
            # reset per student (common in simulators). If you prefer persistent students, remove this reset.
            self.sim.reset(initial_mastery=0.2)
            # If you want to initialize mastery from student pretest, you can inject it here.

            for r in seq:
                kc = self._extract_kc(r, kc_col=kc_col)
                if kc is None:
                    continue
                topic_id = self.kc_to_topic.get(kc, None)
                if topic_id is None:
                    continue

                # action label from dataset signals
                action = self.schema.label_action_from_row(
                    r,
                    cfa_col=cfa_col,
                    duration_col=duration_col,
                    hints_col=hints_col,
                    incorrects_col=incorrects_col,
                )
                action_meta = ActionMeta(action=action, is_tutee=False, force_generation=False)

                # generation_mode proxy from dataset signals
                gen = self.schema.generation_mode_from_row(
                    r,
                    cfa_col=cfa_col,
                    duration_col=duration_col,
                    hints_col=hints_col,
                    incorrects_col=incorrects_col,
                )

                # current state (pre)
                s_pre = self.sim.state.copy()

                # build features
                x_resp = self.sim._build_features(
                    s=s_pre,
                    topic_id=topic_id,
                    action_meta=action_meta,
                    generation_mode=0,  # not used in response stage
                    include_outcome=False,
                )

                cfa = _safe_int(r.get(cfa_col), 0)
                hints = _safe_int(r.get(hints_col), 0)
                incorrects = _safe_int(r.get(incorrects_col), 0)
                duration = _safe_float(r.get(duration_col), 0.0)

                x_trans = self.sim._build_features(
                    s=s_pre,
                    topic_id=topic_id,
                    action_meta=action_meta,
                    generation_mode=gen,
                    include_outcome=True,
                    cfa=cfa,
                    hints=hints,
                    incorrects=incorrects,
                    duration=duration,
                )

                # update deterministic estimator (used only to define delta target for training)
                # Use zero learned delta; estimator update uses EMAs of observations
                # and a minimal mastery update rule derived from observations (CFA).
                # You can replace this with a more sophisticated estimator, but keep it fixed for reproducibility.
                s_post = s_pre.copy()
                self._deterministic_estimator_update(
                    s_post, topic_id=topic_id, cfa=cfa, hints=hints, incorrects=incorrects, duration=duration
                )

                # define delta target as change in latent state fields (not including opp/total_steps bookkeeping)
                delta = np.asarray([
                    float(s_post.mastery[topic_id] - s_pre.mastery[topic_id]),
                    float(s_post.cfa_ema[topic_id] - s_pre.cfa_ema[topic_id]),
                    float(s_post.hint_ema[topic_id] - s_pre.hint_ema[topic_id]),
                    float(s_post.time_ema[topic_id] - s_pre.time_ema[topic_id]),
                    float(s_post.inc_ema[topic_id] - s_pre.inc_ema[topic_id]),
                ], dtype=np.float32)

                # commit post state to builder simulator
                self.sim.state = s_post

                yield TrainingRow(
                    topic_id=topic_id,
                    action_id=int(action),
                    x_resp=x_resp,
                    cfa=int(cfa),
                    x_trans=x_trans,
                    delta=delta,
                    hints=int(hints),
                    incorrects=int(incorrects),
                    duration=float(duration),
                )

    def _deterministic_estimator_update(
        self,
        s: LearnerState,
        *,
        topic_id: int,
        cfa: int,
        hints: int,
        incorrects: int,
        duration: float,
    ) -> None:
        """
        Defines how "state" is computed from KDD observations for training target deltas.

        This is NOT your RL reward and NOT a protégé bonus.
        It is simply the chosen state estimator whose dynamics you then learn with decision trees.

        - mastery is updated as an EMA toward CFA with small dependence on hint/incorrect usage.
          (This is an estimator choice; you can later CV-tune ema_alpha, or even replace with a fitted KT model.)
        """
        alpha = self.ema_alpha

        # bookkeeping
        s.opp[topic_id] += 1
        s.total_steps += 1

        # observational EMAs
        s.cfa_ema[topic_id] = _clip01((1 - alpha) * float(s.cfa_ema[topic_id]) + alpha * float(cfa))
        s.hint_ema[topic_id] = _clip01((1 - alpha) * float(s.hint_ema[topic_id]) + alpha * (float(hints) / 5.0))
        s.time_ema[topic_id] = _clip01((1 - alpha) * float(s.time_ema[topic_id]) + alpha * (float(duration) / 120.0))
        s.inc_ema[topic_id] = _clip01((1 - alpha) * float(s.inc_ema[topic_id]) + alpha * (float(incorrects) / 5.0))

        # mastery estimator update (transparent, bounded, and tied to KDD observables)
        # Interpretation: mastery rises with CFA success, but is moderated by reliance on hints and errors.
        # This is only an estimator used to define learning targets; the trees will learn deltas from features.
        raw_gain = (float(cfa) - 0.5)  # +0.5 if correct, -0.5 if incorrect
        penalty = 0.15 * (float(hints) > 0) + 0.10 * (float(incorrects) > 0)
        gain = raw_gain * (1.0 - penalty)

        # scale by remaining room to improve; keeps bounded and produces diminishing returns
        m = float(s.mastery[topic_id])
        step = 0.08 * gain * (1.0 - m)  # estimator step-size; can be CV-tuned if needed
        s.mastery[topic_id] = _clip01(m + step)


# ----------------------------
# Helper for grouping rows by student (lightweight, no pandas dependency)
# ----------------------------

def group_rows_by_student(
    rows: Iterable[Mapping[str, Any]],
    student_col: str = "Anon Student Id",
    order_key_cols: Optional[Sequence[str]] = None,
) -> List[Tuple[str, List[Mapping[str, Any]]]]:
    """
    Groups rows by student id and sorts within each student by the provided order_key_cols.

    If you load with pandas, you can ignore this and pass already-grouped sequences to KDDTrajectoryBuilder.
    """
    by: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        sid = str(r.get(student_col, "")).strip()
        if sid == "":
            continue
        by.setdefault(sid, []).append(r)

    def _row_key(r: Mapping[str, Any]) -> Tuple:
        if not order_key_cols:
            return (0,)
        key: List[Any] = []
        for c in order_key_cols:
            v = r.get(c)
            # numeric if possible, else string
            try:
                key.append(float(v))
            except Exception:
                key.append(str(v))
        return tuple(key)

    out: List[Tuple[str, List[Mapping[str, Any]]]] = []
    for sid, seq in by.items():
        seq_sorted = sorted(seq, key=_row_key)
        out.append((sid, seq_sorted))
    return out
