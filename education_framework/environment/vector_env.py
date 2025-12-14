
# environment/vector_env.py
from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np

from environment.learner_model import LearnerModel


class VectorLearnerModel:
    """
    Simple vectorized wrapper that runs N independent LearnerModel instances.

    Purpose:
      - Batch inference/training on GPU (high-level + low-level policies)
      - Keep environment logic unchanged (still Python), but amortize agent calls

    Notes:
      - This is not a fully GPU-native env; it is the minimal change that yields
        large GPU utilization by batching policy forward passes and training.
    """

    def __init__(
        self,
        num_envs: int,
        num_topics: int,
        prereqs: Optional[Dict[int, List[int]]] = None,
        seed: int = 123,
    ):
        self.num_envs = int(num_envs)
        self.num_topics = int(num_topics)
        self.envs: List[LearnerModel] = []
        for i in range(self.num_envs):
            self.envs.append(LearnerModel(num_topics=num_topics, prereqs=prereqs, seed=seed + i * 997))

        # cache dimensions
        self.obs_dim = len(self.envs[0].get_observation())

    def reset(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        for i, e in enumerate(self.envs):
            o = e.reset()
            obs[i, :] = np.asarray(o, dtype=np.float32)
        return obs

    def step(
        self,
        modes: np.ndarray,       # bool array, True=tutee, False=tutor
        topics: np.ndarray,      # int array [N]
        ll_actions: List[str],   # list of action strings length N
        done_mask: np.ndarray,   # bool array [N] indicates which envs are already done
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all envs once (those not done).
        Returns (next_obs[N,obs_dim], rewards[N], done[N]).
        """
        next_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = done_mask.copy()

        for i, e in enumerate(self.envs):
            if dones[i]:
                # keep terminal obs as zeros; reward 0
                continue

            t = int(topics[i])
            a = ll_actions[i]
            if bool(modes[i]):  # tutee
                o2, r, d, _ = e.step_tutee(t, a)
            else:
                o2, r, d, _ = e.step_tutor(t, a)

            next_obs[i, :] = np.asarray(o2, dtype=np.float32)
            rewards[i] = float(r)
            dones[i] = bool(d)

        return next_obs, rewards, dones
