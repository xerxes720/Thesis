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

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from education_framework.agents.low_level_agents import avg_layer_cka
import math
import random
from dataclasses import field

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
        v = float(x)
        # IMPORTANT: reject NaN/inf
        if not np.isfinite(v):
            return default
        return v
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
    5..7 are *tutee actions* (not present in KDD; simulated as a metacognitive intervention).

    Keeping |A|=8 preserves comparability with the paper's simulator size.
    """

    # ---- Tutor actions (observed / labelled in KDD) ----
    TUTOR_QUIZ = 0
    TUTOR_HINT = 1
    TUTOR_WORKED_EXAMPLE = 2
    TUTOR_REMEDIATION = 3
    TUTOR_REVIEW = 4

    # ---- Tutee actions (your thesis contribution) ----
    TUTEE_QUIZ = 5  # retrieval prompt
    TUTEE_EXPLAIN = 6  # teach-back / self-explanation
    TUTEE_FIX = 7  # diagnose & fix mistakes


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
        mode = str(getattr(self, "action_mode", "classic"))
        if mode == "discover":
            da = getattr(self, "discovered_action", None)
            if da is None:
                return LowLevelAction.TUTOR_QUIZ

            feat_names = da["feature_names"]
            opp_col = da.get("opp_col", "")
            mu = da["mu"]
            sd = da["sd"]
            centroids = da["centroids"]
            mode_to_action = da["mode_to_action"]

            # build row feature vector in same order
            vals = []
            if "hints" in feat_names:
                vals.append(np.log1p(_safe_int(row.get(hints_col), 0)))
            if "incorrects" in feat_names:
                vals.append(np.log1p(_safe_int(row.get(incorrects_col), 0)))
            if "duration" in feat_names:
                vals.append(np.log1p(max(0.0, _safe_float(row.get(duration_col), 0.0))))
            if "opp" in feat_names:
                opp = _safe_int(row.get(opp_col), 0) if opp_col else 0
                vals.append(np.log1p(opp))

            x = np.asarray(vals, dtype=np.float32)
            xs = (x - mu.reshape(-1)) / sd.reshape(-1)

            d = ((centroids - xs.reshape(1, -1)) ** 2).sum(axis=1)
            mode_id = int(np.argmin(d))
            act_id = int(mode_to_action.get(mode_id, 0))

            return LowLevelAction(act_id)  # assuming enum values match 0..4

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
    total_steps: float = 0.0
    teach_boost: np.ndarray = None  # type: ignore
    last_practice_step: np.ndarray = None  # type: ignore
    retention: np.ndarray = None  # type: ignore

    def __post_init__(self) -> None:
        n = int(self.n_topics)
        self.mastery = np.zeros(n, dtype=np.float32)
        self.opp = np.zeros(n, dtype=np.int32)
        self.cfa_ema = np.zeros(n, dtype=np.float32)
        self.hint_ema = np.zeros(n, dtype=np.float32)
        self.time_ema = np.zeros(n, dtype=np.float32)
        self.inc_ema = np.zeros(n, dtype=np.float32)
        self.total_steps = 0
        self.teach_boost = np.zeros(n, dtype=np.float32)  # new
        self.last_practice_step = np.zeros(n, dtype=np.int32)
        self.retention = np.zeros(n, dtype=np.float32)  # 0..1

    def copy(self) -> "LearnerState":
        s = LearnerState(n_topics=self.n_topics)
        s.mastery = self.mastery.copy()
        s.opp = self.opp.copy()
        s.cfa_ema = self.cfa_ema.copy()
        s.hint_ema = self.hint_ema.copy()
        s.time_ema = self.time_ema.copy()
        s.inc_ema = self.inc_ema.copy()
        s.total_steps = int(self.total_steps)
        s.teach_boost = self.teach_boost.copy()
        s.last_practice_step = self.last_practice_step.copy()
        s.retention = self.retention.copy()

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

    # tutee outcomes estimated from proxy subsets
    # topic_id -> action_id(5..7) -> stats dict
    tutee_outcome_topic: Optional[Dict[int, Dict[int, Dict[str, float]]]] = None

    # topic_id -> leaf_id -> action_id(5..7) -> stats dict
    tutee_outcome_leaf: Optional[Dict[int, Dict[int, Dict[int, Dict[str, float]]]]] = None

    # --- NEW: per-topic beta multipliers (data-driven heterogeneity) ---
    topic_beta_good_mult: Optional[np.ndarray] = None  # shape (n_topics,)
    topic_beta_very_good_mult: Optional[np.ndarray] = None  # shape (n_topics,)
    topic_beta_bad_mult: Optional[np.ndarray] = None  # shape (n_topics,)
    topic_beta_very_bad_mult: Optional[np.ndarray] = None  # shape (n_topics,)

    topic_tutor_action_gain: Optional[np.ndarray] = None

    # Optional: mastery-bin conditioned tutor gains
    tutor_gain_mpre_bins: Optional[np.ndarray] = None            # edges, shape (n_bins+1,)
    topic_tutor_action_gain_by_mpre_bin: Optional[np.ndarray] = None  # shape (n_topics, n_bins, 5)




# ----------------------------
# Learner simulator
# ----------------------------

@dataclass
class KDDLearnerConfig:
    n_topics: int = 7

    mastery_threshold: float = 0.70
    opp_min: int = 1

    # topic_mastery_thresholds: Optional[List[float]] = field(
    #     default_factory=lambda: [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82]
    # )

    # --- Topic heterogeneity / clustering (enables weighted transfer to beat mutual) ---
    # Example for 7 topics: 3 clusters {0,1,2}, {3,4}, {5,6}

    # Per-topic learning difficulty (tutor updates scaled by this)
    # <1.0 = easier, >1.0 = harder. If None, all 1.0

    from dataclasses import field
    # topic_cluster_ids: List[int] = field(default_factory=lambda: [0, 0, 0, 1, 1, 2, 2])
    # topic_difficulty: List[float] = field(default_factory=lambda: [0.85, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40])
    topic_difficulty: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.00, 1.0, 1.0, 1.0, 1.0])
    # --- Action-effect heterogeneity (makes per-topic specialists actually useful) ---
    # Multipliers applied to the mastery-update scale by (topic_cluster, tutor_action_id).
    # tutor_action_id order matches LowLevelAction.TUTOR_*:
    #   [quiz, hint, worked_example, remediation, review]
    # Keep values near 1.0 (e.g., 0.85..1.15) to stay stable.
    enable_action_heterogeneity: bool = False
    # cluster_action_gains: List[List[float]] = field(default_factory=lambda: [
    #     [1.15, 1.05, 0.95, 0.90, 0.95],  # cluster 0: retrieval/hint-heavy
    #     [0.95, 1.00, 1.15, 1.05, 0.95],  # cluster 1: example/remediation-heavy
    #     [0.95, 0.95, 0.90, 1.15, 1.10],  # cluster 2: remediation/review-heavy
    # ])

    # --- tutee (protégé / learning-by-teaching) simulation ---
    # Conservative, bounded mastery bonus with explicit cost.
    # Ablate 0.03 0.06 0.08
    tutee_bonus_base: float = 0.05
    tutee_bonus_mastery_low: float = 0.05
    tutee_bonus_mastery_high: float = 0.95

    # Per-action multipliers (quiz=retrieval, explain=self-explanation, fix=elaboration)
    tutee_mult_quiz: float = 1.0
    tutee_mult_explain: float = 1.4
    tutee_mult_fix: float = 1.4

    # Explicit effort cost in additional "step units" consumed by tutee actions.
    tutee_step_cost_quiz: float = 1.0
    tutee_step_cost_explain: float = 1.1
    tutee_step_cost_fix: float = 1.15

    # Duration multipliers (affects time_ema only; env also consumes step units via step_cost)
    tutee_duration_mult_quiz: float = 1.10
    tutee_duration_mult_explain: float = 1.25
    tutee_duration_mult_fix: float = 1.20

    tutee_ready_quiz: float = 0.40
    tutee_ready_explain: float = 0.50
    tutee_ready_fix: float = 0.55
    tutee_cap_high: float = 0.98

    tutee_reward_lambda: float = 0.0  # start at 0, test 0.1 later

    # step_penalty: float = -0.01
    # correct_reward: float = 0.02
    completion_reward: float = 0.4
    all_done_bonus: float = 1.0

    opp_norm: float = 20.0
    time_norm: float = 120.0
    hints_norm: float = 5.0
    inc_norm: float = 5.0

    # --- tutee -> future tutor learning boost ---
    teach_boost_inc_quiz: float = 0.15
    teach_boost_inc_explain: float = 0.30
    teach_boost_inc_fix: float = 0.25
    teach_boost_decay: float = 0.95  # per step
    teach_boost_max: float = 1.0
    teach_boost_beta_scale: float = 1.50  # tutor update multiplier range: 1 .. 1+0.5

    force_end_on_all_complete: bool = True
    step_penalty: float = 0.0  # start small; tune 0.001..0.01
    # paper_var_weights: List[float] = field(default_factory=lambda: [0.2, 0.7, 0.1])

    # Per-topic observation noise scale (affects neutral noise and/or cfa sampling jitter if you want)
    topic_noise: Optional[List[float]] = None

    # --- Forgetting / spacing ---
    # forget_rate: float = 0.0002      # per "step unit" since last practice
    forget_rate: float = 0.0  # per "step unit" since last practice
    forget_floor: float = 0.1  # don't forget below this baseline mastery
    # retention_from_tutee: float = 0.10  # tutee increases retention (0..1)
    retention_from_tutee: float = 0.0  # tutee increases retention (0..1)
    # retention_decay: float = 0.999  # per step
    retention_decay: float = 1.0  # per step
    retention_init: float = 0.10


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
        # Per-topic one-time completion flags (reset each episode).
        self._topic_completed = np.zeros(self.cfg.n_topics, dtype=np.bool_)

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
    def reset(self, initial_mastery: Union[float, Sequence[float]] = 0.1) -> LearnerState:
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
        self._topic_completed[:] = False
        self.state.teach_boost[:] = 0.0
        self.state.last_practice_step[:] = 0
        self.state.retention[:] = float(self.cfg.retention_init)  # whatever you define as init, e.g. 0.1

        return self.state.copy()

    # def is_done(self) -> bool:
    #     s = self.state
    #     for k in range(self.cfg.n_topics):
    #         if not (float(s.mastery[k]) >= self._topic_threshold(k) and int(s.opp[k]) >= int(self.cfg.opp_min)):
    #             return False
    #     return True
    # def is_done(self) -> bool:
    #     s = self.state
    #     complete = (s.mastery >= self.cfg.mastery_threshold) & (s.opp >= self.cfg.opp_min)
    #     return bool(np.all(complete))
    def is_done(self) -> bool:
        s = self.state
        for k in range(self.cfg.n_topics):
            if not (float(s.mastery[k]) >= self._topic_threshold(k) and int(s.opp[k]) >= int(self.cfg.opp_min)):
                return False
        return True

        # for k in range(self.cfg.n_topics):
        #     if not (float(s.mastery[k]) >= self._topic_threshold(k) and int(s.opp[k]) >= int(self.cfg.opp_min)):
        #         return False
        # return True

    def is_topic_complete(self, topic_id: int) -> bool:
        s = self.state
        return bool(
            (float(s.mastery[topic_id]) >= self._topic_threshold(topic_id)) and
            (int(s.opp[topic_id]) >= int(self.cfg.opp_min))
        )

    def _apply_forgetting_all_except(self, practiced_topic: int, step_cost: float) -> None:
        s = self.state
        cfg = self.cfg

        # decay retention slightly every step
        s.retention *= float(cfg.retention_decay)

        # apply forgetting to all other topics based on time since last practice
        now = int(s.total_steps + max(1, int(step_cost)))
        for k in range(cfg.n_topics):
            if k == practiced_topic:
                continue
            if bool(self._topic_completed[k]):
                continue  # NEW: don't forget completed topics
            dt = max(0, now - int(s.last_practice_step[k]))
            if dt == 0:
                continue

            # retention reduces forgetting (tutee can improve retention)
            r = float(s.retention[k])  # 0..1
            eff_forget = float(cfg.forget_rate) * (1.0 - 0.7 * r)

            m = float(s.mastery[k])
            m2 = m - eff_forget * float(dt) * max(0.0, m - float(cfg.forget_floor))
            s.mastery[k] = _clip01(m2)

    def _tutor_action_gain(self, topic_id: int, action_id: int) -> float:
        """
        Returns an action-effect multiplier for tutor actions (0..4), based on topic cluster.
        For tutee actions (>=5) returns 1.0.
        """
        # TODO
        # return 1.0
        g_bins = getattr(self.bundle, "topic_tutor_action_gain_by_mpre_bin", None)
        edges = getattr(self.bundle, "tutor_gain_mpre_bins", None)

        if g_bins is not None and edges is not None:
            t = int(np.clip(topic_id, 0, g_bins.shape[0] - 1))
            a = int(np.clip(action_id, 0, g_bins.shape[2] - 1))
            m = float(self.state.mastery[t])
            edges = np.asarray(edges, dtype=np.float32)
            n_bins = int(edges.size - 1)
            if n_bins > 0:
                b = int(np.digitize([m], edges[1:-1], right=False)[0]) if n_bins > 1 else 0
                b = int(np.clip(b, 0, n_bins - 1))
                return float(g_bins[t, b, a])

        g = getattr(self.bundle, "topic_tutor_action_gain", None)
        if g is not None:
            t = int(np.clip(topic_id, 0, g.shape[0] - 1))
            a = int(np.clip(action_id, 0, g.shape[1] - 1))
            return float(g[t, a])

        # g = getattr(self.bundle, "topic_tutor_action_gain", None)
        # if g is not None:
        #     t = int(np.clip(topic_id, 0, g.shape[0] - 1))
        #     a = int(np.clip(action_id, 0, g.shape[1] - 1))
        #     return float(g[t, a])
        # 2) Fallback to cluster-based gains if enabled in cfg
        cfg = self.cfg
        if not getattr(cfg, "enable_action_heterogeneity", False):
            return 1.0
        if not getattr(cfg, "topic_cluster_ids", None):
            return 1.0
        cluster_id = int(cfg.topic_cluster_ids[int(topic_id)])
        gains = getattr(cfg, "cluster_action_gains", None)
        if not gains or cluster_id < 0 or cluster_id >= len(gains):
            return 1.0
        row = gains[cluster_id]
        if action_id < 0 or action_id >= len(row):
            return 1.0
        return float(row[action_id])

        # cfg = self.cfg
        # if not getattr(cfg, "enable_action_heterogeneity", False):
        #     return 1.0
        #
        # # only tutor actions 0..4
        # if action_id >= 5:
        #     return 1.0
        #
        # clusters = getattr(cfg, "topic_cluster_ids", None)
        # gains = getattr(cfg, "cluster_action_gains", None)
        # if not clusters or not gains:
        #     return 1.0
        #
        # if topic_id < 0 or topic_id >= len(clusters):
        #     return 1.0
        #
        # cid = int(clusters[topic_id])
        # cid = max(0, min(cid, len(gains) - 1))
        #
        # vec = gains[cid]
        # if not vec or action_id < 0 or action_id >= len(vec):
        #     return 1.0
        #
        # return float(vec[action_id])

    def _topic_scalar(self, arr: Optional[List[float]], topic_id: int, default: float) -> float:
        if not arr:
            return float(default)
        if topic_id < 0 or topic_id >= len(arr):
            return float(default)
        return float(arr[topic_id])

    def _topic_threshold(self, topic_id: int) -> float:
        t = getattr(self.cfg, "topic_mastery_thresholds", None)
        if not t:
            return float(self.cfg.mastery_threshold)
        if topic_id < 0 or topic_id >= len(t):
            return float(self.cfg.mastery_threshold)
        return float(t[topic_id])

    def _beta_mult(self, name: str, topic_id: int) -> float:
        b = self.bundle
        if b is None:
            return 1.0
        arr = getattr(b, name, None)
        if arr is None:
            return 1.0
        try:
            return float(arr[int(topic_id)])
        except Exception:
            return 1.0

    # def _tutor_action_gain(self, topic_id: int, action_id: int) -> float:
    #     if self.bundle is None:
    #         return 1.0
    #     g = getattr(self.bundle, "topic_tutor_action_gain", None)
    #     if g is None:
    #         return 1.0
    #     if action_id < 0 or action_id >= 5:
    #         return 1.0
    #     if topic_id < 0 or topic_id >= g.shape[0]:
    #         return 1.0
    #     return float(g[topic_id, action_id])

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

        x = np.empty(22, dtype=np.float32)

        # ---- base state features (8) ----
        mastery_k = float(s.mastery[topic_id])
        opp_k = float(s.opp[topic_id]) / max(cfg.opp_norm, 1e-6)
        x[0] = mastery_k
        x[1] = opp_k
        x[2] = float(s.cfa_ema[topic_id])
        x[3] = float(s.hint_ema[topic_id])
        x[4] = float(s.time_ema[topic_id])
        x[5] = float(s.inc_ema[topic_id])
        x[6] = float(s.mastery.mean())  # still ok; optional to optimize later
        x[7] = float(s.total_steps) / (cfg.opp_norm * cfg.n_topics)

        # ---- meta (2) ----
        x[8] = 1.0 if action_meta.is_tutee else 0.0
        x[9] = float(generation_mode)

        # ---- one-hot (8) ----
        x[10:18] = 0.0
        a = int(action_meta.action)
        if 0 <= a < 8:
            x[10 + a] = 1.0

        # ---- outcome block (4) ----
        if include_outcome:
            x[18] = float(cfa)
            x[19] = float(hints) / max(cfg.hints_norm, 1e-6)
            x[20] = float(incorrects) / max(cfg.inc_norm, 1e-6)
            x[21] = float(duration) / max(cfg.time_norm, 1e-6)
        else:
            x[18:22] = 0.0

        return x

    def _predict_proba_1(self, model: Any, x: np.ndarray) -> float:
        """
        Robustly return P(y=1) for sklearn-like models.

        Handles degenerate single-class models where predict_proba returns (n, 1)
        and model.classes_ has length 1.
        """
        x2 = x.reshape(1, -1)

        # 1) Try probabilistic path
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(x2, check_input=False)
            except TypeError:
                # some models don't support check_input
                proba = model.predict_proba(x2)

            proba = np.asarray(proba, dtype=np.float32)

            classes = getattr(model, "classes_", None)
            if classes is not None:
                classes = np.asarray(classes)

                # Normal multi-class/binary case: pick the column for class==1 if present
                if proba.ndim == 2 and proba.shape[1] == classes.size:
                    idx_ones = np.where(classes == 1)[0]
                    if idx_ones.size > 0:
                        return _clip01(float(proba[0, int(idx_ones[0])]))

                    # Degenerate: model never saw class 1
                    if classes.size == 1:
                        return 1.0 if int(classes[0]) == 1 else 0.0

                # If classes exist but don't align, fall through to hard prediction

            # If no classes_ info:
            # - if 2+ columns, assume column 1 corresponds to class 1 (common sklearn convention)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return _clip01(float(proba[0, 1]))

            # If only one column and no reliable mapping -> fall back to hard prediction below

        # 2) Hard prediction fallback
        try:
            yhat = model.predict(x2, check_input=False)[0]
        except TypeError:
            yhat = model.predict(x2)[0]
        return _clip01(float(yhat))

    def _predict_aux_int(self, topic_id: int, x: np.ndarray, models: Optional[Dict[int, Any]], default: int) -> int:
        if not models:
            return int(default)
        m = models.get(topic_id)
        if m is None:
            return int(default)
        y = float(m.predict(x.reshape(1, -1), check_input=False)[0])
        return max(0, int(round(y)))

    def _predict_aux_float(self, topic_id: int, x: np.ndarray, models: Optional[Dict[int, Any]],
                           default: float) -> float:
        if not models:
            return float(default)
        m = models.get(topic_id)
        if m is None:
            return float(default)
        y = float(m.predict(x.reshape(1, -1), check_input=False)[0])
        return max(0.0, float(y))

    def _apply_mastery_quality_update(self, topic_id: int, quality: str) -> None:
        s = self.state
        m = float(s.mastery[topic_id])

        params: Optional[MasteryUpdateParams] = None
        if self.bundle is not None and getattr(self.bundle, "quality_bank", None) is not None:
            params = getattr(self.bundle.quality_bank, "mastery_params", None)
        if params is None:
            params = MasteryUpdateParams()  # type: ignore

        if quality == "very_good":
            beta = float(params.beta_very_good) * self._beta_mult("topic_beta_very_good_mult", topic_id)
            m2 = max(m, m + beta * (1.0 - m))
        elif quality == "good":
            beta = float(params.beta_good) * self._beta_mult("topic_beta_good_mult", topic_id)
            m2 = m + beta * (1.0 - m)
        elif quality == "bad":
            beta = float(params.beta_bad) * self._beta_mult("topic_beta_bad_mult", topic_id)
            m2 = m - beta * m
        elif quality == "very_bad":
            beta = float(params.beta_very_bad) * self._beta_mult("topic_beta_very_bad_mult", topic_id)
            m2 = m - beta * m
        else:
            # neutral
            if float(getattr(params, "neutral_noise_std", 0.0)) > 0.0:
                m2 = m + float(self.np_rng.normal(0.0, float(params.neutral_noise_std)))
            else:
                m2 = m

        s.mastery[topic_id] = _clip01(m2)

    from typing import Optional

    def _apply_mastery_quality_update_with_boost(self, topic_id: int, quality: str, boost: float,
                                                 action_id: Optional[int] = None) -> None:

        s = self.state
        m = float(s.mastery[topic_id])

        params: Optional[MasteryUpdateParams] = None
        if self.bundle is not None and getattr(self.bundle, "quality_bank", None) is not None:
            params = getattr(self.bundle.quality_bank, "mastery_params", None)
        if params is None:
            params = MasteryUpdateParams()  # type: ignore

        # scale factor: 1 .. 1 + teach_boost_beta_scale
        # scale = 1.0 + float(self.cfg.teach_boost_beta_scale) * float(boost)

        diff = self._topic_scalar(self.cfg.topic_difficulty, topic_id, 1.0)
        # Harder topic => smaller effective update
        diff_scale = 1.0 / max(0.6, diff)

        # action_scale = 1.0
        # if action_id is not None:
        #     action_scale = self._tutor_action_gain(topic_id, int(action_id))
        scale = diff_scale
        if quality == "very_good" or quality == "good":
            scale = (1.0 + float(self.cfg.teach_boost_beta_scale) * float(boost)) * diff_scale
        # scale = (1.0 + float(self.cfg.teach_boost_beta_scale) * float(boost)) * diff_scale
        if action_id is not None:
            scale *= self._tutor_action_gain(topic_id, int(action_id))

        if quality == "very_good":
            beta = float(params.beta_very_good) * self._beta_mult("topic_beta_very_good_mult", topic_id)
            m2 = max(m, m + beta * scale * (1.0 - m))
        elif quality == "good":
            beta = float(params.beta_good) * self._beta_mult("topic_beta_good_mult", topic_id)
            m2 = m + beta * scale * (1.0 - m)
        elif quality == "bad":
            beta = float(params.beta_bad) * self._beta_mult("topic_beta_bad_mult", topic_id)
            m2 = m - beta * scale * m
        elif quality == "very_bad":
            beta = float(params.beta_very_bad) * self._beta_mult("topic_beta_very_bad_mult", topic_id)
            m2 = m - beta * scale * m
        else:
            m2 = m

        s.mastery[topic_id] = _clip01(m2)

    def _tutee_mastery_bonus(self, mastery: float, a: LowLevelAction) -> float:
        """Conservative, bounded bonus for learning-by-teaching / self-explanation / retrieval.

        This is intentionally *not* learned from KDD (KDD has no explicit tutee interactions).
        """
        cfg = self.cfg
        if mastery < float(cfg.tutee_bonus_mastery_low) or mastery > float(cfg.tutee_bonus_mastery_high):
            return 0.0

        if a == LowLevelAction.TUTEE_QUIZ:
            if mastery < cfg.tutee_ready_quiz or mastery > cfg.tutee_cap_high: return 0.0
            mult = cfg.tutee_mult_quiz
        elif a == LowLevelAction.TUTEE_EXPLAIN:
            if mastery < cfg.tutee_ready_explain or mastery > cfg.tutee_cap_high: return 0.0
            mult = cfg.tutee_mult_explain
        elif a == LowLevelAction.TUTEE_FIX:
            if mastery < cfg.tutee_ready_fix or mastery > cfg.tutee_cap_high: return 0.0
            mult = cfg.tutee_mult_fix
        else:
            mult = 0.0

        base = float(cfg.tutee_bonus_base)
        return max(0.0, base * mult * math.sqrt(1.0 - float(mastery)))

    import math
    import random

    def _apply_tutee_bonus(self, topic_id: int, a: LowLevelAction) -> float:
        """Apply tutee bonus and return the applied delta."""
        s = self.state
        m = float(s.mastery[topic_id])

        # --- 1) readiness (hard gate or soft gate)
        # Hard gate (simple & defendable): return 0.0 if not ready
        if a == LowLevelAction.TUTEE_QUIZ and m < self.cfg.tutee_ready_quiz:
            return 0.0
        if a == LowLevelAction.TUTEE_EXPLAIN and m < self.cfg.tutee_ready_explain:
            return 0.0
        if a == LowLevelAction.TUTEE_FIX and m < self.cfg.tutee_ready_fix:
            return 0.0

        # base magnitude from your existing function
        b = float(self._tutee_mastery_bonus(m, a))

        # --- 2) desirable-difficulty bell (peaks at mid mastery, low at extremes)
        # You can tune center/width; these are conservative defaults.
        center = 0.65
        width = 0.28
        bell = math.exp(-((m - center) / width) ** 2)
        b *= bell

        # --- 3) success-conditioned retrieval (quiz)
        if a == LowLevelAction.TUTEE_QUIZ:
            # probability of successful retrieval increases with mastery
            # simple logistic; tune slope if needed
            k = 10.0
            m0 = 0.45
            p_succ = 1.0 / (1.0 + math.exp(-k * (m - m0)))
            if random.random() > p_succ:
                return 0.0  # failed retrieval => no mastery gain

        # --- 4) fix only helps when there is "something to fix" (struggle signal)
        # if a == LowLevelAction.TUTEE_FIX:
        #     # Use EMAs you already track (values are normalized later; keep it simple)
        #     # If you know these EMAs are in raw units, clamp aggressively.
        #     struggle = float(s.inc_ema[topic_id] + s.hint_ema[topic_id])
        #     struggle = max(0.0, min(1.0, struggle))
        #     struggle = max(struggle, 0.15)  # training floor
        #     b *= struggle
        #     if b <= 0.0:
        #         return 0.0

        # --- 5) keep your diminishing returns (optional)
        # Your original (1-m)*2 is fine; with bell this becomes "mid-mastery sweet spot".
        b *= (1.0 - m) * 2.0

        # --- 6) apply
        if b > 0.0:
            s.mastery[topic_id] = _clip01(m + b)

        return float(b)

    def _apply_observation_updates(
            self,
            topic_id: int,
            *,
            cfa: int,
            hints: int,
            incorrects: int,
            duration: float,
            step_cost: int = 1,
    ) -> None:
        if self.bundle is None:
            alpha = 0.2
        else:
            alpha = float(self.bundle.ema_alpha)

        s = self.state
        s.opp[topic_id] += 1
        s.total_steps += int(max(1, step_cost))

        s.cfa_ema[topic_id] = _clip01((1.0 - alpha) * float(s.cfa_ema[topic_id]) + alpha * float(cfa))
        s.hint_ema[topic_id] = _clip01(
            (1.0 - alpha) * float(s.hint_ema[topic_id]) + alpha * (float(hints) / max(self.cfg.hints_norm, 1e-6)))
        s.time_ema[topic_id] = _clip01(
            (1.0 - alpha) * float(s.time_ema[topic_id]) + alpha * (float(duration) / max(self.cfg.time_norm, 1e-6)))
        s.inc_ema[topic_id] = _clip01(
            (1.0 - alpha) * float(s.inc_ema[topic_id]) + alpha * (float(incorrects) / max(self.cfg.inc_norm, 1e-6)))

    # ---------- core step ----------
    def _paper_vars(self, s: LearnerState) -> np.ndarray:
        """
        Learner variables in [0,1] where higher = better, suitable for percent-change reward.
        Keep small + stable.
        """
        m_mean = float(np.mean(s.mastery))
        m_min = float(np.min(s.mastery))  # forces weakest-topic improvement
        # cov = float(np.mean(np.minimum(s.opp, self.cfg.opp_min) / max(self.cfg.opp_min, 1)))
        # return np.asarray([m_mean, m_min, cov], dtype=np.float32)
        # m = float(np.mean(s.mastery))
        opp_min = max(1, int(self.cfg.opp_min))
        cov = float(np.mean(np.minimum(s.opp, opp_min) / float(opp_min)))
        cfa = float(np.mean(s.cfa_ema))
        #
        # # "less help/struggle/time" is better → invert to keep 'higher is better'
        inv_hint = 1.0 - float(np.mean(s.hint_ema))
        inv_inc = 1.0 - float(np.mean(s.inc_ema))
        inv_time = 1.0 - float(np.mean(s.time_ema))
        #
        return np.asarray([m_mean, m_min, cov, cfa, inv_hint, inv_inc, inv_time], dtype=np.float32)

    # def global_perf_observation(self) -> np.ndarray:
    #     """
    #     Topic-agnostic learner-performance observation (all in [0,1]) for LOW-LEVEL agents.
    #
    #     Matches the paper's treatment: LL agents observe learner performance variables
    #     (score/time/help/engagement-style signals) rather than per-topic latent knowledge vectors.
    #     No topic identity is included; topic context is implied by which LL agent is selected.
    #     """
    #     s = self.state
    #
    #     # Knowledge/score proxies (global)
    #     m_mean = float(np.mean(s.mastery))
    #     m_min = float(np.min(s.mastery))
    #
    #     # Coverage proxy (global exposure)
    #     opp_min = max(1, int(self.cfg.opp_min))
    #     cov = float(np.mean(np.minimum(s.opp, opp_min) / float(opp_min)))
    #
    #     # Global recent performance proxies (higher is better)
    #     cfa = float(np.mean(s.cfa_ema))
    #     inv_hint = 1.0 - float(np.mean(s.hint_ema))
    #     inv_inc = 1.0 - float(np.mean(s.inc_ema))
    #     inv_time = 1.0 - float(np.mean(s.time_ema))
    #
    #     v = np.asarray([m_mean, m_min, cov, cfa, inv_hint, inv_inc, inv_time], dtype=np.float32)
    #
    #     return np.clip(v, 0.0, 1.0)
    #

    def _sample_tutee_outcomes_from_stats(
            self,
            stats: Dict[str, float],
    ) -> Tuple[int, int, int, float]:
        """
        Sample (cfa, hints, incorrects, duration) from precomputed stats.
        Keeps sampling simple and stable.
        """
        # 1) CFA ~ Bernoulli(p)
        p = _clip01(float(stats.get("p_correct", 0.5)))
        cfa = 1 if self.rng.random() < p else 0

        # 2) hints/inc ~ Normal(mean, std) then clamp+round to int >= 0
        def _sample_nonneg_int(mu_key: str, sd_key: str) -> int:
            mu = float(stats.get(mu_key, 0.0))
            sd = float(stats.get(sd_key, 0.0))
            if sd <= 1e-6:
                return max(0, int(round(mu)))
            x = float(self.np_rng.normal(mu, sd))
            return max(0, int(round(x)))

        hints = _sample_nonneg_int("hints_mean", "hints_std")
        incorrects = _sample_nonneg_int("inc_mean", "inc_std")

        # 3) duration ~ Normal(mean, std) then clamp >= 0
        dmu = float(stats.get("dur_mean", 0.0))
        dsd = float(stats.get("dur_std", 0.0))
        if dsd <= 1e-6:
            duration = max(0.0, dmu)
        else:
            duration = max(0.0, float(self.np_rng.normal(dmu, dsd)))

        return cfa, hints, incorrects, duration

    def _refresh_topic_completed_flags(self) -> None:
        for k in range(self.cfg.n_topics):
            if (not bool(self._topic_completed[k])) and self.is_topic_complete(k):
                self._topic_completed[k] = True

    def step(self, topic_id: int, action_meta: ActionMeta) -> Tuple[LearnerState, Dict[str, Any]]:
        if self.bundle is None:
            raise RuntimeError("KDDLearnerModel.bundle is None; provide a trained KDDModelBundle")
        if topic_id < 0 or topic_id >= self.cfg.n_topics:
            raise ValueError(f"topic_id must be in [0, {self.cfg.n_topics - 1}]")

        forced_end = False

        s = self.state

        is_tutee = int(action_meta.action) >= 5

        # Pre-compute x_state / leaf for backoff (only if we have a bank)
        leaf_id = -1
        x_state = None
        qbank = self.bundle.quality_bank
        if qbank is not None:
            x_state = self._state_features(s, topic_id)
            leaf_id = qbank.apply_leaf(topic_id, x_state)

        # -------- (A) OUTCOMES --------
        if (not is_tutee):
            # Tutor actions: unchanged behavior
            x_base = self._build_features(
                s=s,
                topic_id=topic_id,
                action_meta=action_meta,
                generation_mode=0,
                include_outcome=False,
            )

            resp_model = self.bundle.response_models.get(topic_id)
            p_correct = float(s.mastery[topic_id]) if resp_model is None else self._predict_proba_1(resp_model, x_base)

            # NEW: difficulty warps correctness (diff>1 => harder => lower p_correct)
            diff = self._topic_scalar(self.cfg.topic_difficulty, topic_id, 1.0)
            # Use exponent warp: stable + simple + monotonic
            p_correct = _clip01(p_correct ** diff)

            cfa = 1 if self.rng.random() < p_correct else 0

            hints = self._predict_aux_int(topic_id, x_base, self.bundle.hints_models, default=0)
            incorrects = self._predict_aux_int(topic_id, x_base, self.bundle.inc_models, default=(0 if cfa == 1 else 1))
            duration = self._predict_aux_float(topic_id, x_base, self.bundle.time_models,
                                               default=self.bundle.schema.dur_q50)

            # NEW: difficulty scales auxiliary outcomes to widen topic differences
            diff = self._topic_scalar(self.cfg.topic_difficulty, topic_id, 1.0)

            # harder => longer
            duration = float(duration) * (diff ** 0.5)

            # harder => more struggle signals (cap to avoid crazy values)
            hints = int(round(float(hints) * diff))
            hints = min(hints, 10)

            incorrects = int(round(float(incorrects) * diff))
            incorrects = min(incorrects, 10)

            step_cost = 1
            self._apply_forgetting_all_except(topic_id, step_cost=float(step_cost))


        else:
            # Tutee actions: simulated metacognitive intervention.
            # IMPORTANT: we do NOT derive tutee outcomes from KDD (no direct tutee signals)
            # and we do NOT proxy tutee -> tutor for outcome prediction.
            m = float(s.mastery[topic_id])

            # ready = False
            # if action_meta.action == LowLevelAction.TUTEE_EXPLAIN:
            #     ready = (self.cfg.tutee_ready_explain <= m <= self.cfg.tutee_cap_high)
            # elif action_meta.action == LowLevelAction.TUTEE_FIX:
            #     ready = (self.cfg.tutee_ready_fix <= m <= self.cfg.tutee_cap_high)
            # elif action_meta.action == LowLevelAction.TUTEE_QUIZ:
            #     ready = (self.cfg.tutee_ready_quiz <= m <= self.cfg.tutee_cap_high)
            #
            # # Apply quality update based on readiness:
            # if ready:
            #     quality = "good"  # Teaching when ready = good learning
            # else:
            #     quality = "neutral"  # Teaching when not ready = wasted effort
            #
            # boost = float(s.teach_boost[topic_id])
            # self._apply_mastery_quality_update_with_boost(topic_id, quality, boost)
            # Conservative correctness model: no intrinsic boost; the learning gain comes
            # from the tutee mastery bonus below (not from "more correct" outcomes).
            p_correct = _clip01(m)
            cfa = 1 if self.rng.random() < p_correct else 0

            # Keep observational outcomes simple and stable.
            hints = 0
            incorrects = 0 if cfa == 1 else 1

            base_dur = float(self.bundle.schema.dur_q50)
            if action_meta.action == LowLevelAction.TUTEE_EXPLAIN:
                duration = base_dur * float(self.cfg.tutee_duration_mult_explain)
                step_cost = float(self.cfg.tutee_step_cost_explain)
            elif action_meta.action == LowLevelAction.TUTEE_FIX:
                duration = base_dur * float(self.cfg.tutee_duration_mult_fix)
                step_cost = float(self.cfg.tutee_step_cost_fix)
            else:
                duration = base_dur * float(self.cfg.tutee_duration_mult_quiz)
                step_cost = float(self.cfg.tutee_step_cost_quiz)

            self._apply_forgetting_all_except(topic_id, step_cost=float(step_cost))

            # Small noise on duration to avoid degenerate constant signals.
            if duration > 1e-6:
                duration = max(0.0, float(self.np_rng.normal(duration, 0.05 * duration)))

        # 3) generation_mode (optional feature; not required by quality update)
        generation_mode = 1 if (
                    action_meta.force_generation or action_meta.action == LowLevelAction.TUTEE_EXPLAIN) else 0

        # 4) Quality lookup and mastery update
        qbank = self.bundle.quality_bank
        if qbank is None:
            quality = "good" if cfa == 1 else "bad"
            leaf_id = -1
        else:
            # reuse x_state/leaf_id computed above
            if x_state is None:
                x_state = self._state_features(s, topic_id)
            if leaf_id == -1:
                leaf_id = qbank.apply_leaf(topic_id, x_state)
            quality = "neutral" if is_tutee else qbank.predict_quality(
                topic_id=topic_id,
                action_id=int(action_meta.action),
                x_state=x_state,
            )

        v_prev = self._paper_vars(self.state)

        m_before = float(s.mastery[topic_id])

        tutee_bonus = 0.0
        if is_tutee:
            # Neutral quality update (no penalty/reward from tutor-derived leaf qualities)
            # self._apply_mastery_quality_update(topic_id, "good")
            # Mechanistic bounded bonus
            tutee_bonus = self._apply_tutee_bonus(topic_id, action_meta.action)
            # NEW: increase per-topic teach_boost (protégé / self-explanation effect)
            cfg = self.cfg
            if action_meta.action == LowLevelAction.TUTEE_EXPLAIN:
                inc = float(cfg.teach_boost_inc_explain)
            elif action_meta.action == LowLevelAction.TUTEE_FIX:
                inc = float(cfg.teach_boost_inc_fix)
            else:  # TUTEE_QUIZ
                inc = float(cfg.teach_boost_inc_quiz)

            s.teach_boost[topic_id] = min(float(cfg.teach_boost_max), float(s.teach_boost[topic_id]) + inc)
            # NEW: tutee increases retention for this topic (reduces future forgetting)
            s.retention[topic_id] = _clip01(float(s.retention[topic_id]) + float(self.cfg.retention_from_tutee))




        else:
            boost = float(s.teach_boost[topic_id])
            self._apply_mastery_quality_update_with_boost(topic_id, quality, boost, action_id=int(action_meta.action))

        # If topic was already completed earlier this episode, don’t allow mastery to drop
        if bool(self._topic_completed[topic_id]):
            s.mastery[topic_id] = max(float(s.mastery[topic_id]), m_before)

        # --- Intrinsic (topic-local) progress signal for low-level learning ---
        # Keep the paper-style global reward for the HIGH-level agent, but provide a topic-local
        # signal for LOW-level agents to reduce credit-assignment noise in multi-agent HRL.
        m_after = float(s.mastery[topic_id])
        den_local = max(abs(m_before), 0.05)
        r_local = (m_after - m_before) / den_local
        r_local = float(np.clip(r_local, -10, 10))

        # 5) Observational updates (EMAs, opp, steps)
        self._apply_observation_updates(
            topic_id,
            cfa=cfa,
            hints=hints,
            incorrects=incorrects,
            duration=duration,
            step_cost=step_cost,
        )
        self.state.last_practice_step[topic_id] = int(self.state.total_steps)

        # NEW: decay teach_boost over time (short-lived effect)
        s.teach_boost[topic_id] *= float(self.cfg.teach_boost_decay)

        # 6) Reward shaping (optional)
        done = self.is_done()
        if self.cfg.force_end_on_all_complete and all(self._topic_completed):
            done = True
            forced_end = True
        # reward = float(self.cfg.step_penalty + (self.cfg.correct_reward if cfa == 1 else 0.0) + (self.cfg.completion_reward if done else 0.0))
        v_next = self._paper_vars(self.state)
        den = np.maximum(np.abs(v_prev), 0.05)  # critical: avoid dividing by ~0
        pct = (v_next - v_prev) / den
        r_step = float(np.mean(pct))
        # w = np.asarray(getattr(self.cfg, "paper_var_weights", [1.0] * int(pct.shape[0])), dtype=np.float32)
        # if w.size != pct.size:
        #     r_step = float(np.mean(pct))
        #
        # else:
        #     ws = float(np.sum(w))
        #     if ws <= 1e-8:
        #         r_step = float(np.mean(pct))
        #     else:
        #         w = w / ws
        #         r_step = float(np.dot(w, pct))
        # optional but recommended for stability
        r_step = float(np.clip(r_step, -10, 10))

        # reward = r_step + (self.cfg.completion_reward if done else 0.0)
        # One-time completion reward per topic/subtask:
        # give +r_c the first time this topic reaches (mastery >= threshold AND opp >= opp_min).
        topic_completion_bonus = 0.0
        if (not bool(self._topic_completed[topic_id])) and self.is_topic_complete(topic_id):
            self._topic_completed[topic_id] = True
            topic_completion_bonus = float(self.cfg.completion_reward)
            # info["topic_completion_bonus"] = float(topic_completion_bonus)
            # info["topic_completed"] = bool(self._topic_completed[topic_id])

        # reward = r_step + topic_completion_bonus
        # # can add a lambda 0.5 or 1 or 2 to tutee bonud -> lambda * tutee_bonus
        # reward += float(self.cfg.tutee_reward_lambda) * tutee_bonus
        # reward -= float(self.cfg.step_penalty) * float(step_cost)

        reward_global = r_step + topic_completion_bonus
        reward_global += float(self.cfg.tutee_reward_lambda) * tutee_bonus

        reward_local_total = r_local + topic_completion_bonus
        reward_local_total += float(self.cfg.tutee_reward_lambda) * tutee_bonus

        info = {
            "p_correct": p_correct,
            "cfa": cfa,
            "hints": hints,
            "incorrects": incorrects,
            "duration": duration,
            "step_cost": float(step_cost),
            "generation_mode": generation_mode,
            "quality": quality,
            "leaf": leaf_id,
            "tutee_bonus": float(tutee_bonus),
            # "reward": reward,
            "reward_global": float(reward_global),
            "reward_local": float(reward_local_total),
            "reward": float(reward_global),
            "done": done,
            "forced_end": forced_end,
            "diff": float(self._topic_scalar(self.cfg.topic_difficulty, topic_id, 1.0)),
            "topic_threshold": float(self._topic_threshold(topic_id)),
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
        self.sim.reset(initial_mastery=0.1)

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

    # learner_model.py
    from typing import Callable, Optional, Any, Mapping, Iterable, Tuple, List

    # inside KDDTrajectoryBuilder
    def iter_training_rows(
            self,
            rows_by_student: Iterable[Tuple[str, List[Mapping[str, Any]]]],
            *,
            kc_col: str = "KC(Default)",
            cfa_col: str = "Correct First Attempt",
            duration_col: str = "Step Duration (sec)",
            hints_col: str = "Hints",
            incorrects_col: str = "Incorrects",
            action_labeler: Optional[Callable[..., int]] = None,
    ) -> Iterable[TrainingRow]:

        for _sid, seq in rows_by_student:
            self.sim.reset(initial_mastery=0.1)

            for r in seq:
                kc = self._extract_kc(r, kc_col)
                if kc is None:
                    continue
                topic_id = self.kc_to_topic.get(kc)
                if topic_id is None:
                    continue

                # ---- outcomes (available in KDD row) ----
                cfa = _safe_int(r.get(cfa_col), 0)
                hints = _safe_int(r.get(hints_col), 0)
                incorrects = _safe_int(r.get(incorrects_col), 0)
                duration = _safe_float(r.get(duration_col), 0.0)

                s_pre = self.sim.state.copy()
                mastery_pre = float(s_pre.mastery[topic_id])
                x_state = self.sim._state_features(s_pre, topic_id)

                # ---- action id (schema OR external labeler) ----
                if action_labeler is None:
                    action = self.schema.label_tutor_action_from_row(
                        r,
                        cfa_col=cfa_col,
                        duration_col=duration_col,
                        hints_col=hints_col,
                        incorrects_col=incorrects_col,
                    )
                else:
                    action = int(action_labeler(
                        row=r,
                        topic_id=int(topic_id),
                        s_pre=s_pre,
                        x_state=x_state,
                        cfa=int(cfa),
                        hints=int(hints),
                        incorrects=int(incorrects),
                        duration=float(duration),
                    ))

                action_meta = ActionMeta(action=action, is_tutee=False, force_generation=False)

                # x_resp must be built with the *chosen* action id
                x_resp = self.sim._build_features(
                    s=s_pre,
                    topic_id=topic_id,
                    action_meta=action_meta,
                    generation_mode=0,
                    include_outcome=False,
                )

                # deterministic estimator update (unchanged)
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
        # step = 0.08 * gain * (1.0 - m)
        step = 0.04 * gain * (1.0 - m)
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
