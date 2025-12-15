
# environment/async_vector_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import multiprocessing as mp

from environment.learner_model import LearnerModel


@dataclass
class AsyncVecConfig:
    num_workers: int = 8
    envs_per_worker: int = 32
    seed: int = 123


def _worker_loop(
    conn,
    worker_id: int,
    num_topics: int,
    envs_per_worker: int,
    prereqs: Optional[Dict[int, List[int]]],
    seed: int,
    tutor_actions: List[str],
    tutee_actions: List[str],
):
    # Create env instances in the worker process.
    envs: List[LearnerModel] = []
    for i in range(envs_per_worker):
        envs.append(LearnerModel(num_topics=num_topics, prereqs=prereqs, seed=seed + worker_id * 100_000 + i * 997))

    obs_dim = len(envs[0].get_observation())

    while True:
        msg = conn.recv()
        cmd = msg[0]

        if cmd == "close":
            conn.close()
            return

        if cmd == "get_obs_dim":
            conn.send(obs_dim)
            continue

        if cmd == "reset":
            obs = np.zeros((envs_per_worker, obs_dim), dtype=np.float32)
            for i, e in enumerate(envs):
                o = e.reset()
                obs[i, :] = np.asarray(o, dtype=np.float32)
            conn.send(obs)
            continue

        if cmd == "step":
            # payload: modes(bool)[B], topics(int)[B], ll_action_idx(int)[B], done(bool)[B]
            modes, topics, ll_action_idx, done_mask = msg[1], msg[2], msg[3], msg[4]
            next_obs = np.zeros((envs_per_worker, obs_dim), dtype=np.float32)
            rewards = np.zeros((envs_per_worker,), dtype=np.float32)
            dones = done_mask.copy()

            for i, e in enumerate(envs):
                if dones[i]:
                    continue

                t = int(topics[i])
                if bool(modes[i]):  # tutee
                    a_str = tutee_actions[int(ll_action_idx[i])]
                    o2, r, d, _ = e.step_tutee(t, a_str)
                else:
                    a_str = tutor_actions[int(ll_action_idx[i])]
                    o2, r, d, _ = e.step_tutor(t, a_str)

                next_obs[i, :] = np.asarray(o2, dtype=np.float32)
                rewards[i] = float(r)
                dones[i] = bool(d)

            conn.send((next_obs, rewards, dones))
            continue

        raise RuntimeError(f"Unknown command: {cmd}")


class AsyncVectorLearnerModel:
    """
    Multiprocess vector environment to reduce main-process CPU bottleneck
    and keep the GPU fed with batched inference/training.

    - Windows-safe: uses spawn.
    - Each worker hosts envs_per_worker independent LearnerModel instances.
    - Main process sends batched actions and receives batched transitions.
    """

    def __init__(
        self,
        num_topics: int,
        prereqs: Optional[Dict[int, List[int]]] = None,
        cfg: Optional[AsyncVecConfig] = None,
        tutor_actions: Optional[List[str]] = None,
        tutee_actions: Optional[List[str]] = None,
    ):
        self.num_topics = int(num_topics)
        self.prereqs = prereqs or {}
        self.cfg = cfg or AsyncVecConfig()

        self.tutor_actions = tutor_actions or ["hint", "worked_example", "reflection_question", "no_help"]
        self.tutee_actions = tutee_actions or ["ask_explanation", "ask_worked_example", "ask_summary", "show_mistake_and_ask_fix"]

        # Start workers
        self.ctx = mp.get_context("spawn")
        self.num_workers = int(self.cfg.num_workers)
        self.envs_per_worker = int(self.cfg.envs_per_worker)
        self.num_envs = self.num_workers * self.envs_per_worker

        self.parents = []
        self.procs = []

        for wid in range(self.num_workers):
            parent_conn, child_conn = self.ctx.Pipe()
            proc = self.ctx.Process(
                target=_worker_loop,
                args=(child_conn, wid, self.num_topics, self.envs_per_worker, self.prereqs, self.cfg.seed, self.tutor_actions, self.tutee_actions),
                daemon=True,
            )
            proc.start()
            self.parents.append(parent_conn)
            self.procs.append(proc)

        # Get obs_dim from worker 0
        self.parents[0].send(("get_obs_dim",))
        self.obs_dim = int(self.parents[0].recv())

    def close(self):
        for p in self.parents:
            try:
                p.send(("close",))
            except Exception:
                pass
        for proc in self.procs:
            try:
                proc.join(timeout=2.0)
            except Exception:
                pass

    def reset(self) -> np.ndarray:
        for p in self.parents:
            p.send(("reset",))
        chunks = [p.recv() for p in self.parents]
        return np.concatenate(chunks, axis=0)

    def step(
        self,
        modes: np.ndarray,         # bool [N]
        topics: np.ndarray,        # int [N]
        ll_action_idx: np.ndarray, # int [N]
        done_mask: np.ndarray,     # bool [N]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Split into worker chunks
        next_chunks = []
        reward_chunks = []
        done_chunks = []

        for w, p in enumerate(self.parents):
            s = w * self.envs_per_worker
            e = s + self.envs_per_worker
            p.send(("step", modes[s:e], topics[s:e], ll_action_idx[s:e], done_mask[s:e]))

        for p in self.parents:
            nxt, rew, dn = p.recv()
            next_chunks.append(nxt)
            reward_chunks.append(rew)
            done_chunks.append(dn)

        next_obs = np.concatenate(next_chunks, axis=0)
        rewards = np.concatenate(reward_chunks, axis=0)
        dones = np.concatenate(done_chunks, axis=0)
        return next_obs, rewards, dones
