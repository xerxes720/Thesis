# education_framework/environment/learner_model.py

"""learner_model.py

KDD Algebra learner simulator (case-study implementation).

This version implements a *paper-style* transition model:
  - a routing decision tree partitions learner state into leaves
  - each leaf stores an action -> quality category mapping
  - mastery is updated using category-based beta constants (no learned EMA deltas)

Why this matters
----------------
- It aligns the simulator narrative with the original paper (leaf-level action
  quality categories).
- It removes the "EMA double update" failure mode (learning EMAs as deltas and
  then EMA-updating again from outcomes).

What is learned from KDD
------------------------
- Response model per topic: P(CFA=1) given current state + action
- Optional auxiliary outcome models per topic: hints, incorrects, duration
- QualityTreeBank per topic: routing tree + leaf/action quality categories

What is deterministic at runtime
--------------------------------
- EMAs are updated once from realized outcomes
- Opportunity counts and total steps are bookkeeping

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import math
import random

import numpy as np

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    # Optional dependency in runtime; required when using a trained QualityTreeBank.
    from education_framework.models.quality_tree_bank import QualityTreeBank, MasteryUpdateParams
except Exception:  # pragma: no cover
    QualityTreeBank = Any  # type: ignore
    MasteryUpdateParams = Any  # type: ignore


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


# ----------------------------
# Action space (8 ids, with explicit tutor vs tutee split)
# ----------------------------

class LowLevelAction(IntEnum):
    """Unified action id space used by the simulator.

    0..4 are tutor-labelled actions (learned directly from KDD via schema rules).
    5..7 are *tutee actions* (not present in KDD as interventions; calibrated via proxies).

    Keeping |A|=8 preserves comparability with the paper's simulator size.
    """

    # ---- Tutor actions (observed / labelled in KDD) ----
    TUTOR_QUIZ = 0
    TUTOR_HINT = 1
    TUTOR_WORKED_EXAMPLE = 2
    TUTOR_REMEDIATION = 3
    TUTOR_REVIEW = 4

    # ---- Tutee actions (your thesis contribution) ----
    TUTEE_QUIZ = 5        # retrieval prompt
    TUTEE_EXPLAIN = 6     # teach-back / self-explanation
    TUTEE_FIX = 7         # diagnose & fix mistakes


@dataclass(frozen=True)
class ActionMeta:
    action: LowLevelAction
    is_tutee: bool = False
    force_generation: bool = False


# ----------------------------
# Dataset-driven labeling schema
# ----------------------------

@dataclass
class KDDActionSchema:
    """Quantile thresholds used for simple, transparent labeling and proxies."""

    # duration quantiles (sec)
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

        for i, r in enumerate(rows):
            if i >= max_rows:
                break
            durs.append(_safe_float(r.get(duration_col), 0.0))
            hints.append(_safe_float(r.get(hints_col), 0.0))
            incs.append(_safe_float(r.get(incorrects_col), 0.0))

        schema = cls()
        schema.dur_q25 = cls._quantile(durs, 0.25, schema.dur_q25)
        schema.dur_q50 = cls._quantile(durs, 0.50, schema.dur_q50)
        schema.dur_q75 = cls._quantile(durs, 0.75, schema.dur_q75)
        schema.dur_q90 = cls._quantile(durs, 0.90, schema.dur_q90)

        schema.hints_q50 = cls._quantile(hints, 0.50, schema.hints_q50)
        schema.hints_q75 = cls._quantile(hints, 0.75, schema.hints_q75)

        schema.inc_q50 = cls._quantile(incs, 0.50, schema.inc_q50)
        schema.inc_q75 = cls._quantile(incs, 0.75, schema.inc_q75)

        return schema

    # ---- Tutor action labeling (from KDD observables) ----
    def label_tutor_action_from_row(
        self,
        row: Mapping[str, Any],
        *,
        cfa_col: str = "Correct First Attempt",
        duration_col: str = "Step Duration (sec)",
        hints_col: str = "Hints",
        incorrects_col: str = "Incorrects",
    ) -> LowLevelAction:
        """Label a KDD step into one of the *tutor* actions (0..4).

        This is intentionally simple and threshold-based so it is explainable.
        """
        cfa = _safe_int(row.get(cfa_col), 0)
        hints = _safe_int(row.get(hints_col), 0)
        inc = _safe_int(row.get(incorrects_col), 0)
        dur = _safe_float(row.get(duration_col), 0.0)

        # remediation: observable struggle
        if (cfa == 0) or (inc > 0) or (inc >= int(self.inc_q75) + 1):
            return LowLevelAction.TUTOR_REMEDIATION

        # hint-heavy: scaffolding
        if hints > 0:
            return LowLevelAction.TUTOR_HINT

        # slow but clean: review / deeper processing
        if (dur >= self.dur_q75) and (hints <= self.hints_q50) and (inc <= self.inc_q50):
            return LowLevelAction.TUTOR_REVIEW

        # fast and correct: quiz / fluency
        if (dur <= self.dur_q25) and (cfa == 1):
            return LowLevelAction.TUTOR_QUIZ

        # default: worked-example-style guidance then practice
        return LowLevelAction.TUTOR_WORKED_EXAMPLE

    # ---- Simple generative-mode proxy (optional feature) ----
    def generation_mode_from_row(
        self,
        row: Mapping[str, Any],
        *,
        cfa_col: str = "Correct First Attempt",
        duration_col: str = "Step Duration (sec)",
        hints_col: str = "Hints",
        incorrects_col: str = "Incorrects",
    ) -> int:
        cfa = _safe_int(row.get(cfa_col), 0)
        hints = _safe_int(row.get(hints_col), 0)
        inc = _safe_int(row.get(incorrects_col), 0)
        dur = _safe_float(row.get(duration_col), 0.0)
        return int((cfa == 1) and (hints <= self.hints_q50) and (inc <= self.inc_q50) and (dur >= self.dur_q50))


# ----------------------------
# Learner state
# ----------------------------

@dataclass
class LearnerState:
    n_topics: int
    mastery: np.ndarray = None  # type: ignore
    opp: np.ndarray = None  # type: ignore
    cfa_ema: np.ndarray = None  # type: ignore
    hint_ema: np.ndarray = None  # type: ignore
    time_ema: np.ndarray = None  # type: ignore
    inc_ema: np.ndarray = None  # type: ignore
    total_steps: int = 0

    def __post_init__(self) -> None:
        n = int(self.n_topics)
        self.mastery = np.zeros(n, dtype=np.float32)
        self.opp = np.zeros(n, dtype=np.int32)
        self.cfa_ema = np.zeros(n, dtype=np.float32)
        self.hint_ema = np.zeros(n, dtype=np.float32)
        self.time_ema = np.zeros(n, dtype=np.float32)
        self.inc_ema = np.zeros(n, dtype=np.float32)
        self.total_steps = 0

    def copy(self) -> "LearnerState":
        s = LearnerState(n_topics=self.n_topics)
        s.mastery = self.mastery.copy()
        s.opp = self.opp.copy()
        s.cfa_ema = self.cfa_ema.copy()
        s.hint_ema = self.hint_ema.copy()
        s.time_ema = self.time_ema.copy()
        s.inc_ema = self.inc_ema.copy()
        s.total_steps = int(self.total_steps)
        return s


# ----------------------------
# Model bundle (trained elsewhere)
# ----------------------------

@dataclass
class KDDModelBundle:
    n_topics: int
    schema: KDDActionSchema

    # maps raw KC string -> topic id 0..n_topics-1
    kc_to_topic: Dict[str, int]

    response_models: Dict[int, Any]

    # optional outcome models
    hints_models: Optional[Dict[int, Any]] = None
    time_models: Optional[Dict[int, Any]] = None
    inc_models: Optional[Dict[int, Any]] = None

    # paper-style transition model
    quality_bank: Optional[QualityTreeBank] = None

    one_hot_actions: bool = True
    ema_alpha: float = 0.2


# ----------------------------
# Learner simulator
# ----------------------------

@dataclass
class KDDLearnerConfig:
    n_topics: int = 7

    mastery_threshold: float = 0.85
    opp_min: int = 3

    step_penalty: float = -0.01
    correct_reward: float = 0.02
    completion_reward: float = 3.0

    opp_norm: float = 20.0
    time_norm: float = 120.0
    hints_norm: float = 5.0
    inc_norm: float = 5.0


class KDDLearnerModel:
    """KDD-based learner simulator used by your HRL framework."""

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
            raise RuntimeError("joblib not available")
        joblib.dump({"cfg": self.cfg, "bundle": self.bundle}, path)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> "KDDLearnerModel":
        if joblib is None:
            raise RuntimeError("joblib not available")
        obj = joblib.load(path)
        return cls(cfg=obj["cfg"], bundle=obj["bundle"], seed=seed)

    # ---------- env-like API ----------
    def reset(self, initial_mastery: Union[float, Sequence[float]] = 0.2) -> LearnerState:
        if isinstance(initial_mastery, (list, tuple, np.ndarray)):
            arr = np.asarray(initial_mastery, dtype=np.float32)
            if arr.shape[0] != self.cfg.n_topics:
                raise ValueError("initial_mastery length must equal n_topics")
            self.state.mastery[:] = arr
        else:
            self.state.mastery[:] = float(initial_mastery)
        self.state.opp[:] = 0
        self.state.cfa_ema[:] = 0.0
        self.state.hint_ema[:] = 0.0
        self.state.time_ema[:] = 0.0
        self.state.inc_ema[:] = 0.0
        self.state.total_steps = 0
        return self.state.copy()

    def is_done(self) -> bool:
        s = self.state
        mastered = (s.mastery >= self.cfg.mastery_threshold).astype(np.int32)
        enough_opp = (s.opp >= self.cfg.opp_min).astype(np.int32)
        return bool(np.all((mastered * enough_opp) > 0))

    # ---------- feature engineering ----------
    def _state_features(self, s: LearnerState, topic_id: int) -> np.ndarray:
        """Features used by the routing quality tree: state only (no action, no outcomes)."""
        cfg = self.cfg
        mastery_k = float(s.mastery[topic_id])
        opp_k = float(s.opp[topic_id]) / max(cfg.opp_norm, 1e-6)
        cfa_ema_k = float(s.cfa_ema[topic_id])
        hint_ema_k = float(s.hint_ema[topic_id])
        time_ema_k = float(s.time_ema[topic_id])
        inc_ema_k = float(s.inc_ema[topic_id])
        global_mastery = float(np.mean(s.mastery))
        total_steps_norm = float(s.total_steps) / (cfg.opp_norm * cfg.n_topics)
        return np.asarray(
            [mastery_k, opp_k, cfa_ema_k, hint_ema_k, time_ema_k, inc_ema_k, global_mastery, total_steps_norm],
            dtype=np.float32,
        )

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
        base = self._state_features(s, topic_id).tolist()
        is_tutee = 1.0 if action_meta.is_tutee else 0.0
        gen = float(generation_mode)
        feats: List[float] = base + [is_tutee, gen]

        # one-hot action (fixed length 8)
        a = int(action_meta.action)
        one_hot = [0.0] * 8
        one_hot[a] = 1.0
        feats.extend(one_hot)

        if include_outcome:
            feats.extend(
                [
                    float(cfa),
                    float(hints) / max(cfg.hints_norm, 1e-6),
                    float(incorrects) / max(cfg.inc_norm, 1e-6),
                    float(duration) / max(cfg.time_norm, 1e-6),
                ]
            )
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        return np.asarray(feats, dtype=np.float32)

    def _predict_proba_1(self, model: Any, x: np.ndarray) -> float:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x.reshape(1, -1))
            return float(proba[0, 1])
        y = float(model.predict(x.reshape(1, -1))[0])
        return _clip01(y)

    def _predict_aux_int(self, topic_id: int, x: np.ndarray, models: Optional[Dict[int, Any]], default: int) -> int:
        if not models:
            return int(default)
        m = models.get(topic_id)
        if m is None:
            return int(default)
        y = float(m.predict(x.reshape(1, -1))[0])
        return max(0, int(round(y)))

    def _predict_aux_float(self, topic_id: int, x: np.ndarray, models: Optional[Dict[int, Any]], default: float) -> float:
        if not models:
            return float(default)
        m = models.get(topic_id)
        if m is None:
            return float(default)
        y = float(m.predict(x.reshape(1, -1))[0])
        return max(0.0, float(y))

    def _outcome_proxy_action(self, a: LowLevelAction) -> LowLevelAction:
        """Map tutee actions to the closest tutor action for outcome prediction.

        Rationale: response/aux models are trained on tutor-labelled KDD actions (0..4).
        """
        if a == LowLevelAction.TUTEE_QUIZ:
            return LowLevelAction.TUTOR_QUIZ
        if a == LowLevelAction.TUTEE_EXPLAIN:
            return LowLevelAction.TUTOR_HINT
        if a == LowLevelAction.TUTEE_FIX:
            return LowLevelAction.TUTOR_REMEDIATION
        return a

    def _apply_mastery_quality_update(self, topic_id: int, quality: str) -> None:
        s = self.state
        m = float(s.mastery[topic_id])

        params: Optional[MasteryUpdateParams] = None
        if self.bundle is not None and getattr(self.bundle, "quality_bank", None) is not None:
            params = getattr(self.bundle.quality_bank, "mastery_params", None)
        if params is None:
            params = MasteryUpdateParams()  # type: ignore

        if quality == "very_good":
            m2 = max(m, float(params.mastery_jump))
        elif quality == "good":
            m2 = m + float(params.beta_good) * (1.0 - m)
        elif quality == "bad":
            m2 = m - float(params.beta_bad) * m
        elif quality == "very_bad":
            m2 = m - float(params.beta_very_bad) * m
        else:
            # neutral
            if float(getattr(params, "neutral_noise_std", 0.0)) > 0.0:
                m2 = m + float(self.np_rng.normal(0.0, float(params.neutral_noise_std)))
            else:
                m2 = m

        s.mastery[topic_id] = _clip01(m2)

    def _apply_observation_updates(self, topic_id: int, *, cfa: int, hints: int, incorrects: int, duration: float) -> None:
        if self.bundle is None:
            alpha = 0.2
        else:
            alpha = float(self.bundle.ema_alpha)

        s = self.state
        s.opp[topic_id] += 1
        s.total_steps += 1

        s.cfa_ema[topic_id] = _clip01((1.0 - alpha) * float(s.cfa_ema[topic_id]) + alpha * float(cfa))
        s.hint_ema[topic_id] = _clip01((1.0 - alpha) * float(s.hint_ema[topic_id]) + alpha * (float(hints) / max(self.cfg.hints_norm, 1e-6)))
        s.time_ema[topic_id] = _clip01((1.0 - alpha) * float(s.time_ema[topic_id]) + alpha * (float(duration) / max(self.cfg.time_norm, 1e-6)))
        s.inc_ema[topic_id] = _clip01((1.0 - alpha) * float(s.inc_ema[topic_id]) + alpha * (float(incorrects) / max(self.cfg.inc_norm, 1e-6)))

    # ---------- core step ----------
    def step(self, topic_id: int, action_meta: ActionMeta) -> Tuple[LearnerState, Dict[str, Any]]:
        if self.bundle is None:
            raise RuntimeError("KDDLearnerModel.bundle is None; provide a trained KDDModelBundle")
        if topic_id < 0 or topic_id >= self.cfg.n_topics:
            raise ValueError(f"topic_id must be in [0, {self.cfg.n_topics - 1}]")

        s = self.state

        # For outcome prediction, use a tutor-action proxy for tutee actions.
        proxy_action = self._outcome_proxy_action(action_meta.action)
        proxy_meta = ActionMeta(action=proxy_action, is_tutee=action_meta.is_tutee, force_generation=action_meta.force_generation)

        x_base = self._build_features(
            s=s,
            topic_id=topic_id,
            action_meta=proxy_meta,
            generation_mode=0,
            include_outcome=False,
        )

        # 1) Sample CFA
        resp_model = self.bundle.response_models.get(topic_id)
        p_correct = float(s.mastery[topic_id]) if resp_model is None else self._predict_proba_1(resp_model, x_base)
        cfa = 1 if self.rng.random() < p_correct else 0

        # 2) Sample auxiliary outcomes
        hints = self._predict_aux_int(topic_id, x_base, self.bundle.hints_models, default=0)
        incorrects = self._predict_aux_int(topic_id, x_base, self.bundle.inc_models, default=(0 if cfa == 1 else 1))
        duration = self._predict_aux_float(topic_id, x_base, self.bundle.time_models, default=self.bundle.schema.dur_q50)

        # Simple tutee outcome shaping (kept minimal; avoids destabilizing the simulator)
        if action_meta.action == LowLevelAction.TUTEE_QUIZ:
            hints = min(hints, int(self.bundle.schema.hints_q50))
        elif action_meta.action == LowLevelAction.TUTEE_EXPLAIN:
            hints = min(hints, int(self.bundle.schema.hints_q50))
            duration = max(duration, float(self.bundle.schema.dur_q75))
        elif action_meta.action == LowLevelAction.TUTEE_FIX:
            duration = max(duration, float(self.bundle.schema.dur_q50))

        # 3) generation_mode (optional feature; not required by quality update)
        generation_mode = 1 if (action_meta.force_generation or action_meta.action == LowLevelAction.TUTEE_EXPLAIN) else 0

        # 4) Quality lookup and mastery update
        qbank = self.bundle.quality_bank
        if qbank is None:
            quality = "good" if cfa == 1 else "bad"
            leaf_id = -1
        else:
            x_state = self._state_features(s, topic_id)
            quality = qbank.predict_quality(topic_id=topic_id, action_id=int(action_meta.action), x_state=x_state)
            leaf_id = qbank.apply_leaf(topic_id, x_state)

        self._apply_mastery_quality_update(topic_id, quality)

        # 5) Observational updates (EMAs, opp, steps)
        self._apply_observation_updates(topic_id, cfa=cfa, hints=hints, incorrects=incorrects, duration=duration)

        # 6) Reward shaping (optional)
        done = self.is_done()
        reward = float(self.cfg.step_penalty + (self.cfg.correct_reward if cfa == 1 else 0.0) + (self.cfg.completion_reward if done else 0.0))

        info = {
            "p_correct": p_correct,
            "cfa": cfa,
            "hints": hints,
            "incorrects": incorrects,
            "duration": duration,
            "generation_mode": generation_mode,
            "quality": quality,
            "leaf": leaf_id,
            "reward": reward,
            "done": done,
        }
        return self.state.copy(), info


# ----------------------------
# Trajectory builder for training
# ----------------------------

@dataclass
class TrainingRow:
    topic_id: int
    action_id: int  # tutor actions 0..4
    x_resp: np.ndarray
    x_state: np.ndarray
    mastery_pre: float
    cfa: int
    hints: int
    incorrects: int
    duration: float
    delta_mastery: float


class KDDTrajectoryBuilder:
    """Extract supervised rows from KDD for training response/aux models and quality trees."""

    def __init__(
        self,
        n_topics: int,
        kc_to_topic: Mapping[str, int],
        schema: KDDActionSchema,
        ema_alpha: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.n_topics = int(n_topics)
        self.kc_to_topic = dict(kc_to_topic)
        self.schema = schema
        self.ema_alpha = float(ema_alpha)
        self.rng = random.Random(seed)

        self.sim = KDDLearnerModel(
            cfg=KDDLearnerConfig(n_topics=self.n_topics),
            bundle=KDDModelBundle(
                n_topics=self.n_topics,
                schema=schema,
                kc_to_topic=dict(kc_to_topic),
                response_models={},
                quality_bank=None,
                one_hot_actions=True,
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
        if not s:
            return None
        if "~~" in s:
            s = s.split("~~")[0].strip()
        return s if s else None

    def iter_training_rows(
        self,
        rows_by_student: Iterable[Tuple[str, List[Mapping[str, Any]]]],
        *,
        kc_col: str = "KC(Default)",
        cfa_col: str = "Correct First Attempt",
        duration_col: str = "Step Duration (sec)",
        hints_col: str = "Hints",
        incorrects_col: str = "Incorrects",
    ) -> Iterable[TrainingRow]:

        for _sid, seq in rows_by_student:
            self.sim.reset(initial_mastery=0.2)

            for r in seq:
                kc = self._extract_kc(r, kc_col)
                if kc is None:
                    continue
                topic_id = self.kc_to_topic.get(kc)
                if topic_id is None:
                    continue

                action = self.schema.label_tutor_action_from_row(
                    r,
                    cfa_col=cfa_col,
                    duration_col=duration_col,
                    hints_col=hints_col,
                    incorrects_col=incorrects_col,
                )
                action_meta = ActionMeta(action=action, is_tutee=False, force_generation=False)

                s_pre = self.sim.state.copy()
                mastery_pre = float(s_pre.mastery[topic_id])

                x_resp = self.sim._build_features(
                    s=s_pre,
                    topic_id=topic_id,
                    action_meta=action_meta,
                    generation_mode=0,
                    include_outcome=False,
                )
                x_state = self.sim._state_features(s_pre, topic_id)

                cfa = _safe_int(r.get(cfa_col), 0)
                hints = _safe_int(r.get(hints_col), 0)
                incorrects = _safe_int(r.get(incorrects_col), 0)
                duration = _safe_float(r.get(duration_col), 0.0)

                # Deterministic estimator update defines delta_mastery used for quality calibration
                s_post = s_pre.copy()
                self._deterministic_estimator_update(
                    s_post,
                    topic_id=topic_id,
                    cfa=cfa,
                    hints=hints,
                    incorrects=incorrects,
                    duration=duration,
                )
                delta_mastery = float(s_post.mastery[topic_id] - s_pre.mastery[topic_id])

                # commit
                self.sim.state = s_post

                yield TrainingRow(
                    topic_id=int(topic_id),
                    action_id=int(action),
                    x_resp=x_resp,
                    x_state=x_state,
                    mastery_pre=mastery_pre,
                    cfa=int(cfa),
                    hints=int(hints),
                    incorrects=int(incorrects),
                    duration=float(duration),
                    delta_mastery=delta_mastery,
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
        """Defines a simple, fixed state estimator for building training targets.

        This is not an RL reward and not a tutee bonus.
        It is only used to extract a consistent delta_mastery signal from KDD.
        """
        alpha = float(self.ema_alpha)

        s.opp[topic_id] += 1
        s.total_steps += 1

        s.cfa_ema[topic_id] = _clip01((1.0 - alpha) * float(s.cfa_ema[topic_id]) + alpha * float(cfa))
        s.hint_ema[topic_id] = _clip01((1.0 - alpha) * float(s.hint_ema[topic_id]) + alpha * (float(hints) / 5.0))
        s.time_ema[topic_id] = _clip01((1.0 - alpha) * float(s.time_ema[topic_id]) + alpha * (float(duration) / 120.0))
        s.inc_ema[topic_id] = _clip01((1.0 - alpha) * float(s.inc_ema[topic_id]) + alpha * (float(incorrects) / 5.0))

        # mastery estimator: bounded, diminishing returns
        m = float(s.mastery[topic_id])
        raw_gain = (float(cfa) - 0.5)  # +0.5 if correct, -0.5 if incorrect
        penalty = 0.15 * (float(hints) > 0) + 0.10 * (float(incorrects) > 0)
        gain = raw_gain * (1.0 - penalty)
        step = 0.08 * gain * (1.0 - m)
        s.mastery[topic_id] = _clip01(m + step)


# ----------------------------
# Helper: group rows by student
# ----------------------------

def group_rows_by_student(
    rows: Iterable[Mapping[str, Any]],
    student_col: str = "Anon Student Id",
    order_key_cols: Optional[Sequence[str]] = None,
) -> List[Tuple[str, List[Mapping[str, Any]]]]:
    by: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        sid = str(r.get(student_col, "")).strip()
        if sid:
            by.setdefault(sid, []).append(r)

    def _row_key(r: Mapping[str, Any]) -> Tuple:
        if not order_key_cols:
            return (0,)
        key: List[Any] = []
        for c in order_key_cols:
            v = r.get(c)
            try:
                key.append(float(v))
            except Exception:
                key.append(str(v))
        return tuple(key)

    out: List[Tuple[str, List[Mapping[str, Any]]]] = []
    for sid, seq in by.items():
        out.append((sid, sorted(seq, key=_row_key)))
    return out
