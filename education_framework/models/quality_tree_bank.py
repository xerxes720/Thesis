# -----------------------------
# Model container (saved to disk)
# -----------------------------
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor


@dataclass
class LeafQualityModel:
    tree: DecisionTreeRegressor
    leaf_to_action_quality: Dict[int, Dict[str, str]]
    leaf_to_action_delta: Dict[int, Dict[str, np.ndarray]]   # NEW
    default_quality: str = "neutral"
    default_delta: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))


class QualityTreeBank:
    def __init__(self, num_topics: int, feature_names: List[str]):
        self.num_topics = num_topics
        self.feature_names = feature_names
        self.bank: Dict[int, LeafQualityModel] = {}  # topic_id -> model

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "num_topics": self.num_topics,
            "feature_names": self.feature_names,
            "bank": self.bank,
            "version": 1,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "QualityTreeBank":
        payload = joblib.load(path)
        qt = QualityTreeBank(payload["num_topics"], payload["feature_names"])
        qt.bank = payload["bank"]
        return qt

    def predict_quality(self, topic_id: int, action: str, x: np.ndarray) -> str:
        m = self.bank.get(topic_id)
        if m is None:
            return "neutral"
        leaf = int(m.tree.apply(x.reshape(1, -1))[0])
        leaf_map = m.leaf_to_action_quality.get(leaf) or {}
        return leaf_map.get(action, m.default_quality)

    def predict_delta(self, topic_id: int, action: str, x: np.ndarray) -> np.ndarray:
        m = self.bank.get(topic_id)
        if m is None:
            return np.zeros(5, dtype=np.float32)
        leaf = int(m.tree.apply(x.reshape(1, -1))[0])
        leaf_map = m.leaf_to_action_delta.get(leaf) or {}
        d = leaf_map.get(action, None)
        if d is None:
            return np.zeros(5, dtype=np.float32)
        return d