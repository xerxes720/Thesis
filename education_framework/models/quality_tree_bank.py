# education_framework/models/quality_tree_bank.py
"""quality_tree_bank.py

Paper-style transition model for the KDD case study.

Key idea
--------
A *routing tree* partitions the learner state space into leaves. At each leaf,
we store a mapping from action_id -> quality category in:
    {"very_bad", "bad", "neutral", "good", "very_good"}

At runtime:
  leaf = routing_tree.apply(x_state)
  quality = leaf_to_action_quality[leaf][action_id]
  mastery <- mastery_update(quality, betas)

This makes action effects explicit and is easy to explain in the thesis.

Notes
-----
- This module intentionally does NOT model EMA deltas. EMAs should be updated
  deterministically from realized outcomes (once) to avoid double-counting.
- The routing tree can be a classifier or regressor; only its leaf partition is
  used at inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import joblib

# sklearn is only needed when building / applying trees
from sklearn.tree import DecisionTreeClassifier


QUALITY_LEVELS: List[str] = ["very_bad", "bad", "neutral", "good", "very_good"]


@dataclass
class MasteryUpdateParams:
    """Parameters for category-based mastery updates."""

    mastery_jump: float = 0.95  # used for very_good

    # Positive learning rates (applied as + beta * (1 - M))
    beta_good: float = 0.04
    beta_very_good: float = 0.07

    # Negative decay rates (applied as - beta * M)
    beta_bad: float = 0.02
    beta_very_bad: float = 0.05

    # Optional small noise for neutral category (applied in mastery space)
    neutral_noise_std: float = 0.0


@dataclass
class LeafQualityModel:
    """Per-topic routing tree + leaf/action quality table."""

    tree: Any  # typically sklearn DecisionTreeClassifier

    # leaf_id -> {action_id -> quality_str}
    leaf_to_action_quality: Dict[int, Dict[int, str]] = field(default_factory=dict)

    # (optional) leaf_id -> {action_id -> mean_score}
    leaf_to_action_score_mean: Dict[int, Dict[int, float]] = field(default_factory=dict)

    default_quality: str = "neutral"


class QualityTreeBank:
    """Holds LeafQualityModel for each topic."""

    def __init__(
        self,
        num_topics: int,
        feature_names: List[str],
        mastery_params: Optional[MasteryUpdateParams] = None,
    ) -> None:
        self.num_topics = int(num_topics)
        self.feature_names = list(feature_names)
        self.mastery_params = mastery_params or MasteryUpdateParams()
        self.bank: Dict[int, LeafQualityModel] = {}

    # ---------- IO ----------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "num_topics": self.num_topics,
            "feature_names": self.feature_names,
            "mastery_params": self.mastery_params,
            "bank": self.bank,
            "version": 2,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "QualityTreeBank":
        payload = joblib.load(path)
        qt = QualityTreeBank(
            num_topics=int(payload["num_topics"]),
            feature_names=list(payload["feature_names"]),
            mastery_params=payload.get("mastery_params") or MasteryUpdateParams(),
        )
        qt.bank = payload["bank"]
        return qt

    # ---------- Inference ----------
    def _get_model(self, topic_id: int) -> Optional[LeafQualityModel]:
        return self.bank.get(int(topic_id))

    def apply_leaf(self, topic_id: int, x_state: "Any") -> int:
        m = self._get_model(topic_id)
        if m is None:
            return -1
        leaf = int(m.tree.apply(x_state.reshape(1, -1))[0])
        return leaf

    def predict_quality(self, topic_id: int, action_id: int, x_state: "Any") -> str:
        m = self._get_model(topic_id)
        if m is None:
            return "neutral"
        leaf = int(m.tree.apply(x_state.reshape(1, -1))[0])
        amap = m.leaf_to_action_quality.get(leaf)
        if not amap:
            return m.default_quality
        q = amap.get(int(action_id))
        return q if q in QUALITY_LEVELS else m.default_quality

    def predict_score_mean(self, topic_id: int, action_id: int, x_state: "Any") -> Optional[float]:
        m = self._get_model(topic_id)
        if m is None:
            return None
        leaf = int(m.tree.apply(x_state.reshape(1, -1))[0])
        smap = m.leaf_to_action_score_mean.get(leaf)
        if not smap:
            return None
        return float(smap.get(int(action_id))) if int(action_id) in smap else None
