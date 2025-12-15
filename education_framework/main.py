
# main_async_batched.py
import numpy as np
import torch
import multiprocessing as mp

from agents.high_level_agent import HighLevelAgent, HighLevelAgentConfig
from agents.low_level_agents import TutorLowLevelAgent, TuteeLowLevelAgent, LowLevelAgentConfig
from environment.async_vector_env import AsyncVectorLearnerModel, AsyncVecConfig


def create_agents(num_topics: int, use_tutee: bool = True):
    hl_cfg = HighLevelAgentConfig(num_topics=num_topics, use_tutee=use_tutee)
    ll_cfg = LowLevelAgentConfig(num_topics=num_topics)
    hl = HighLevelAgent(hl_cfg)
    tutors = [TutorLowLevelAgent(ll_cfg) for _ in range(num_topics)]
    tutee = TuteeLowLevelAgent(ll_cfg) if use_tutee else None
    return hl, tutors, tutee


def train_async(
    num_topics: int = 8,
    use_tutee: bool = True,
    num_workers: int = 8,
    envs_per_worker: int = 32,  # total envs = workers * envs_per_worker
    learner_batches: int = 24,   # your previous run had 24 batches; keep comparable
    log_every_batches: int = 1,
):
    prereqs = {
        1: [0],
        2: [1],
        3: [2],
        4: [2],
        5: [2],
        6: [5],
        7: [6],
    }

    tutor_actions = ["hint", "worked_example", "reflection_question", "no_help"]
    tutee_actions = ["ask_explanation", "ask_worked_example", "ask_summary", "show_mistake_and_ask_fix"]

    env = AsyncVectorLearnerModel(
        num_topics=num_topics,
        prereqs=prereqs,
        cfg=AsyncVecConfig(num_workers=num_workers, envs_per_worker=envs_per_worker, seed=123),
        tutor_actions=tutor_actions,
        tutee_actions=tutee_actions,
    )

    hl, tutors, tutee = create_agents(num_topics=num_topics, use_tutee=use_tutee)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("device:", device)
    print("num_envs:", env.num_envs, "obs_dim:", env.obs_dim)

    # Force-initialize networks so you can verify device placement
    hl._ensure_networks(input_dim=env.obs_dim)
    for t in range(num_topics):
        tutors[t]._ensure_networks(input_dim=env.obs_dim + 1)  # + topic feature
    if use_tutee and tutee is not None:
        tutee._ensure_networks(input_dim=env.obs_dim + 1)

    print("HL policy device:", next(hl.policy_net.parameters()).device)
    print("LL policy device:", next(tutors[0].policy_net.parameters()).device)

    # Epsilon schedule
    eps_start, eps_end = 0.2, 0.0
    total_batches = int(learner_batches)

    try:
        for batch_idx in range(1, total_batches + 1):
            progress = min(1.0, batch_idx / total_batches)
            eps = eps_start + (eps_end - eps_start) * progress

            hl.set_epsilon(eps)
            for a in tutors:
                a.set_epsilon(eps)
            if use_tutee and tutee is not None:
                tutee.set_epsilon(eps)

            obs = env.reset()  # [N, obs_dim]
            done = np.zeros((env.num_envs,), dtype=bool)

            total_reward = np.zeros((env.num_envs,), dtype=np.float32)
            steps = 0

            topic_counts = np.zeros((num_topics,), dtype=np.int64)
            tutor_action_counts = {a: 0 for a in tutor_actions}
            tutee_action_counts = {a: 0 for a in tutee_actions}

            while not done.all():
                obs_t = torch.from_numpy(obs).to(device=device, non_blocking=True)

                # High-level batched (GPU)
                hl_actions = hl.select_action_batch(obs_t)  # [N]
                hl_modes_t, hl_topics_t = hl.decode_actions_batch(hl_actions)
                hl_modes = hl_modes_t.detach().cpu().numpy().astype(bool)
                hl_topics = hl_topics_t.detach().cpu().numpy().astype(np.int64)

                # Low-level action index per env (single int array)
                ll_action_idx = np.zeros((env.num_envs,), dtype=np.int64)

                # Tutors grouped by topic
                for t in range(num_topics):
                    idx = np.where((~hl_modes) & (hl_topics == t) & (~done))[0]
                    if idx.size == 0:
                        continue
                    topic_col = torch.full((idx.size, 1), float(t), device=device)
                    ll_obs = torch.cat([obs_t[idx], topic_col], dim=1)

                    a_idx = tutors[t].select_action_batch(ll_obs).detach().cpu().numpy().astype(np.int64)
                    ll_action_idx[idx] = a_idx

                    # logging
                    topic_counts[t] += idx.size
                    for k in a_idx.tolist():
                        tutor_action_counts[tutor_actions[int(k)]] += 1

                # Tutee group (all topics mixed)
                if use_tutee and tutee is not None:
                    idx = np.where((hl_modes) & (~done))[0]
                    if idx.size > 0:
                        topic_col = torch.from_numpy(hl_topics[idx]).to(device=device).float().unsqueeze(1)
                        ll_obs = torch.cat([obs_t[idx], topic_col], dim=1)
                        a_idx = tutee.select_action_batch(ll_obs).detach().cpu().numpy().astype(np.int64)
                        ll_action_idx[idx] = a_idx

                        for k in a_idx.tolist():
                            tutee_action_counts[tutee_actions[int(k)]] += 1
                        for t in hl_topics[idx]:
                            topic_counts[int(t)] += 1

                # Step envs in parallel (CPU across workers)
                next_obs, rewards, done2 = env.step(hl_modes, hl_topics, ll_action_idx, done)

                # Batched updates (CPU->replay, replay->GPU inside agents)
                active = np.where(~done)[0]
                if active.size > 0:
                    actions_cpu = hl_actions.detach().cpu()

                    hl.update_batch(
                        obs_cpu=torch.from_numpy(obs[active]),
                        actions_cpu=actions_cpu[active],
                        rewards_cpu=torch.from_numpy(rewards[active]),
                        next_obs_cpu=torch.from_numpy(next_obs[active]),
                        dones_cpu=torch.from_numpy(done2[active].astype(np.float32)),
                    )

                    # Tutor updates by topic
                    for t in range(num_topics):
                        idx = np.where((~hl_modes) & (hl_topics == t) & (~done))[0]
                        if idx.size == 0:
                            continue
                        ll_obs = np.concatenate([obs[idx], np.full((idx.size, 1), float(t), dtype=np.float32)], axis=1)
                        ll_next = np.concatenate([next_obs[idx], np.full((idx.size, 1), float(t), dtype=np.float32)], axis=1)
                        tutors[t].update_batch(
                            obs_cpu=torch.from_numpy(ll_obs),
                            actions_cpu=torch.from_numpy(ll_action_idx[idx]),
                            rewards_cpu=torch.from_numpy(rewards[idx]),
                            next_obs_cpu=torch.from_numpy(ll_next),
                            dones_cpu=torch.from_numpy(done2[idx].astype(np.float32)),
                        )

                    # Tutee updates
                    if use_tutee and tutee is not None:
                        idx = np.where((hl_modes) & (~done))[0]
                        if idx.size > 0:
                            ll_obs = np.concatenate([obs[idx], hl_topics[idx].astype(np.float32).reshape(-1, 1)], axis=1)
                            ll_next = np.concatenate([next_obs[idx], hl_topics[idx].astype(np.float32).reshape(-1, 1)], axis=1)
                            tutee.update_batch(
                                obs_cpu=torch.from_numpy(ll_obs),
                                actions_cpu=torch.from_numpy(ll_action_idx[idx]),
                                rewards_cpu=torch.from_numpy(rewards[idx]),
                                next_obs_cpu=torch.from_numpy(ll_next),
                                dones_cpu=torch.from_numpy(done2[idx].astype(np.float32)),
                            )

                total_reward[~done] += rewards[~done]
                obs = next_obs
                done = done2
                steps += 1

            # Logging
            if (batch_idx % log_every_batches) == 0:
                mean_reward = float(np.mean(total_reward))
                print(f"[Batch {batch_idx:3d}/{total_batches}] eps={eps:.3f} steps={steps} mean_reward={mean_reward:.4f}")
                total_topic = int(topic_counts.sum()) or 1
                print("  Topic choice frequencies:")
                for t in range(num_topics):
                    print(f"    - Topic {t}: {topic_counts[t] / total_topic * 100:5.1f}%")
                total_tutor = sum(tutor_action_counts.values()) or 1
                print("  Tutor action frequencies:")
                for a in tutor_actions:
                    print(f"    - {a:20s}: {tutor_action_counts[a] / total_tutor * 100:5.1f}%")
                if use_tutee and tutee is not None:
                    total_tutee = sum(tutee_action_counts.values()) or 1
                    print("  Tutee action frequencies:")
                    for a in tutee_actions:
                        print(f"    - {a:20s}: {tutee_action_counts[a] / total_tutee * 100:5.1f}%")
                print()

        print("Training finished.")
    finally:
        env.close()


if __name__ == "__main__":
    mp.freeze_support()
    train_async()
