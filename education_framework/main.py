# main.py  (SIMPLE BATCHED, NO ASYNC/VECTOR_ENV, GPU-FRIENDLY)
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

from agents.high_level_agent import HighLevelAgent, HighLevelAgentConfig
from agents.low_level_agents import TutorLowLevelAgent, TuteeLowLevelAgent, LowLevelAgentConfig
from environment.learner_model import LearnerModel


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    num_topics: int = 8
    use_tutee: bool = True

    # parallelism (GPU utilization lever)
    num_envs: int = 128

    # episode settings
    episodes: int = 500
    max_steps: int = 200

    # epsilon schedule
    eps_start: float = 0.2
    eps_end: float = 0.0

    # logging
    log_every: int = 1

    # reproducibility
    seed: int = 123

    # performance toggles
    use_tf32: bool = True
    matmul_precision: str = "high"  # "high" | "medium" | (torch may ignore on older versions)


TUTOR_ACTIONS = ["hint", "worked_example", "reflection_question", "no_help"]
TUTEE_ACTIONS = ["ask_explanation", "ask_worked_example", "ask_summary", "show_mistake_and_ask_fix"]


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_envs(num_envs: int, num_topics: int, prereqs: Optional[Dict[int, List[int]]], seed: int) -> List[LearnerModel]:
    envs: List[LearnerModel] = []
    for i in range(num_envs):
        envs.append(LearnerModel(num_topics=num_topics, prereqs=prereqs, seed=seed + i * 997))
    return envs


def create_agents(num_topics: int, use_tutee: bool) -> Tuple[HighLevelAgent, List[TutorLowLevelAgent], Optional[TuteeLowLevelAgent]]:
    hl_cfg = HighLevelAgentConfig(num_topics=num_topics, use_tutee=use_tutee)
    ll_cfg = LowLevelAgentConfig(num_topics=num_topics)

    hl = HighLevelAgent(hl_cfg)
    tutors = [TutorLowLevelAgent(ll_cfg) for _ in range(num_topics)]
    tutee = TuteeLowLevelAgent(ll_cfg) if use_tutee else None
    return hl, tutors, tutee


def _to_2d_long(x: torch.Tensor) -> torch.Tensor:
    # ensure shape [B, 1] and dtype long
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return x.long()


def train(cfg: TrainConfig) -> None:
    set_global_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        if cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision(cfg.matmul_precision)
        except Exception:
            pass

    # Simple prereq chain (edit to match your curriculum graph)
    prereqs = {
        1: [0],
        2: [1],
        3: [2],
        4: [2],
        5: [2],
        6: [5],
        7: [6],
    }

    envs = make_envs(cfg.num_envs, cfg.num_topics, prereqs=prereqs, seed=cfg.seed)

    hl, tutors, tutee = create_agents(cfg.num_topics, cfg.use_tutee)

    # If your agent constructors do not auto-move models to GPU, do it here
    # (Only if those attributes exist in your implementation.)
    for a in [hl] + tutors + ([tutee] if tutee is not None else []):
        if a is None:
            continue

        policy = getattr(a, "policy_net", None)
        if policy is not None:
            policy.to(device)

        target = getattr(a, "target_net", None)
        if target is not None:
            target.to(device)

    # Determine observation size by one reset
    obs0 = envs[0].reset()
    obs_dim = len(obs0)

    # Main training loop
    for ep in range(1, cfg.episodes + 1):
        # reset all envs
        obs = np.zeros((cfg.num_envs, obs_dim), dtype=np.float32)
        done = np.zeros((cfg.num_envs,), dtype=np.bool_)

        for i, e in enumerate(envs):
            obs[i] = np.asarray(e.reset(), dtype=np.float32)

        ep_return = np.zeros((cfg.num_envs,), dtype=np.float32)

        # epsilon linear schedule across episodes
        progress = (ep - 1) / max(1, (cfg.episodes - 1))
        eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * progress

        hl.set_epsilon(eps)
        for a in tutors:
            a.set_epsilon(eps)
        if tutee is not None:
            tutee.set_epsilon(eps)

        for step in range(cfg.max_steps):
            active_mask = ~done
            if not active_mask.any():
                break

            # obs tensor on GPU for batched inference
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)

            # 1) High-level batched action selection
            with torch.no_grad():
                hl_actions = hl.select_action_batch(obs_t)
            hl_actions_cpu = _to_2d_long(hl_actions).detach().cpu()

            # decode HL into (mode, topic)
            # expected: hl_modes is boolean array (True=tutee, False=tutor), hl_topics is int array [0..T-1]
            hl_modes, hl_topics = hl.decode_actions_batch(hl_actions_cpu)
            hl_modes = np.asarray(hl_modes, dtype=np.bool_).reshape(-1)
            hl_topics = np.asarray(hl_topics, dtype=np.int64).reshape(-1)

            # 2) Low-level batched action selection
            ll_action_idx = np.zeros((cfg.num_envs,), dtype=np.int64)

            # Tutor groups: per topic, batch on GPU
            tutor_indices_by_topic: List[np.ndarray] = []
            for t in range(cfg.num_topics):
                idx = np.where(active_mask & (~hl_modes) & (hl_topics == t))[0]
                tutor_indices_by_topic.append(idx)

                if idx.size == 0:
                    continue

                topic_col = torch.full((idx.size, 1), float(t), device=device, dtype=torch.float32)
                ll_obs = torch.cat([obs_t[idx], topic_col], dim=1)

                with torch.no_grad():
                    a_idx = tutors[t].select_action_batch(ll_obs)

                a_idx = a_idx.detach().cpu().numpy().astype(np.int64).reshape(-1)
                ll_action_idx[idx] = a_idx

            # Tutee group: all topics mixed, batch once on GPU
            tutee_idx = np.where(active_mask & hl_modes)[0]
            if (tutee is not None) and (tutee_idx.size > 0):
                topic_col = torch.as_tensor(hl_topics[tutee_idx], device=device, dtype=torch.float32).view(-1, 1)
                ll_obs = torch.cat([obs_t[tutee_idx], topic_col], dim=1)

                with torch.no_grad():
                    a_idx = tutee.select_action_batch(ll_obs)

                a_idx = a_idx.detach().cpu().numpy().astype(np.int64).reshape(-1)
                ll_action_idx[tutee_idx] = a_idx

            # 3) Step environments (simple loop; policies/training are batched)
            next_obs = obs.copy()
            rewards = np.zeros((cfg.num_envs,), dtype=np.float32)
            done2 = done.copy()

            active_idx = np.where(active_mask)[0]
            for i in active_idx:
                topic = int(hl_topics[i])
                if hl_modes[i] and (tutee is not None):
                    action = TUTEE_ACTIONS[int(ll_action_idx[i])]
                    o2, r, d, _info = envs[i].step_tutee(topic_id=topic, tutee_action=action)
                else:
                    action = TUTOR_ACTIONS[int(ll_action_idx[i])]
                    o2, r, d, _info = envs[i].step_tutor(topic_id=topic, tutor_action=action)

                next_obs[i] = np.asarray(o2, dtype=np.float32)
                rewards[i] = float(r)
                done2[i] = bool(d)

            ep_return += rewards
            done = done2

            # 4) Batched updates (CPU tensors; agent can move/sample internally)
            # High-level update over active transitions
            obs_cpu = torch.from_numpy(obs[active_idx])
            next_obs_cpu = torch.from_numpy(next_obs[active_idx])
            r_cpu = torch.from_numpy(rewards[active_idx])
            d_cpu = torch.from_numpy(done[active_idx].astype(np.float32))
            a_hl_cpu = hl_actions_cpu[active_idx].view(-1).long()  # already [B,1] long

            hl.update_batch(obs_cpu, a_hl_cpu, r_cpu, next_obs_cpu, d_cpu)

            # Low-level updates: per tutor topic
            for t in range(cfg.num_topics):
                idx = tutor_indices_by_topic[t]
                if idx.size == 0:
                    continue

                # low-level state includes topic column
                topic_col_cpu = torch.full((idx.size, 1), float(t), dtype=torch.float32)
                ll_obs_cpu = torch.cat([torch.from_numpy(obs[idx]), topic_col_cpu], dim=1)
                ll_next_obs_cpu = torch.cat([torch.from_numpy(next_obs[idx]), topic_col_cpu], dim=1)

                a_ll_cpu = torch.from_numpy(ll_action_idx[idx]).long()
                r_ll_cpu = torch.from_numpy(rewards[idx])
                d_ll_cpu = torch.from_numpy(done[idx].astype(np.float32))

                tutors[t].update_batch(ll_obs_cpu, a_ll_cpu, r_ll_cpu, ll_next_obs_cpu, d_ll_cpu)

            # Low-level update: tutee
            if (tutee is not None) and (tutee_idx.size > 0):
                topic_col_cpu = torch.as_tensor(hl_topics[tutee_idx], dtype=torch.float32).view(-1, 1)
                ll_obs_cpu = torch.cat([torch.from_numpy(obs[tutee_idx]), topic_col_cpu], dim=1)
                ll_next_obs_cpu = torch.cat([torch.from_numpy(next_obs[tutee_idx]), topic_col_cpu], dim=1)

                a_ll_cpu = torch.from_numpy(ll_action_idx[tutee_idx]).long()
                r_ll_cpu = torch.from_numpy(rewards[tutee_idx])
                d_ll_cpu = torch.from_numpy(done[tutee_idx].astype(np.float32))

                tutee.update_batch(ll_obs_cpu, a_ll_cpu, r_ll_cpu, ll_next_obs_cpu, d_ll_cpu)

            # advance
            obs = next_obs

        # Logging (use observation slices; avoids touching env internals)
        # obs layout: mastery_learner[0..T-1], mastery_tutee[0..T-1], ...
        mL = float(obs[:, 0:cfg.num_topics].mean())
        mT = float(obs[:, cfg.num_topics:2 * cfg.num_topics].mean())
        mean_ret = float(ep_return.mean())

        if (ep % cfg.log_every) == 0:
            print(f"[Episode {ep:4d}] eps={eps:.3f}  mean_return={mean_ret:+.4f}  mean_mastery(L)={mL:.3f}  mean_mastery(T)={mT:.3f}")

    print("Training finished.")


if __name__ == "__main__":
    cfg = TrainConfig(
        num_topics=8,
        use_tutee=True,
        num_envs=128,     # increase to 256/512 if GPU is underutilized and you have RAM
        episodes=500,
        max_steps=200,
        log_every=1,
        seed=123,
    )
    train(cfg)
