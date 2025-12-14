
# main_batched.py
import numpy as np
import torch

from agents.high_level_agent import HighLevelAgent, HighLevelAgentConfig
from agents.low_level_agents import TutorLowLevelAgent, TuteeLowLevelAgent, LowLevelAgentConfig
from environment.vector_env import VectorLearnerModel


def create_agents(num_topics: int, use_tutee: bool = True):
    hl_cfg = HighLevelAgentConfig(num_topics=num_topics, use_tutee=use_tutee)
    ll_cfg = LowLevelAgentConfig(num_topics=num_topics)

    high_level_agent = HighLevelAgent(hl_cfg)
    tutor_agents = [TutorLowLevelAgent(ll_cfg) for _ in range(num_topics)]
    tutee_agent = TuteeLowLevelAgent(ll_cfg) if use_tutee else None
    return high_level_agent, tutor_agents, tutee_agent


def train_vectorized(
    num_topics: int = 8,
    use_tutee: bool = True,
    num_envs: int = 256,
    learner_batches: int = 6000 // 256 + 1,  # approx same "episode count"
    log_every_batches: int = 10,
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

    env = VectorLearnerModel(num_envs=num_envs, num_topics=num_topics, prereqs=prereqs)
    hl, tutors, tutee = create_agents(num_topics=num_topics, use_tutee=use_tutee)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("device:", device)
    print("num_envs:", num_envs, "obs_dim:", env.obs_dim)

    tutor_action_names = tutors[0].get_action_meanings()
    tutor_action_to_idx = {a: i for i, a in enumerate(tutor_action_names)}
    tutee_action_names = tutee.get_action_meanings() if tutee is not None else []
    tutee_action_to_idx = {a: i for i, a in enumerate(tutee_action_names)}

    # Epsilon schedule across batches
    eps_start, eps_end = 0.2, 0.0
    total_batches = int(learner_batches)

    for batch_idx in range(1, total_batches + 1):
        progress = min(1.0, batch_idx / total_batches)
        eps = eps_start + (eps_end - eps_start) * progress
        hl.set_epsilon(eps)
        for a in tutors:
            a.set_epsilon(eps)
        if tutee is not None:
            tutee.set_epsilon(eps)

        obs = env.reset()  # [N, obs_dim] float32
        done = np.zeros((num_envs,), dtype=bool)

        total_reward = np.zeros((num_envs,), dtype=np.float32)
        steps = 0

        # Logging counters
        topic_counts = np.zeros((num_topics,), dtype=np.int64)
        tutor_action_counts = {a: 0 for a in tutor_action_names}
        tutee_action_counts = {a: 0 for a in tutee_action_names}

        while not done.all():
            obs_t = torch.from_numpy(obs).to(device=device, non_blocking=True)

            # High-level batched
            hl_actions = hl.select_action_batch(obs_t)  # [N] int64 on GPU
            hl_modes_t, hl_topics_t = hl.decode_actions_batch(hl_actions)

            hl_modes = hl_modes_t.detach().cpu().numpy().astype(bool)
            hl_topics = hl_topics_t.detach().cpu().numpy().astype(np.int64)

            # Low-level decisions stored as both index and string
            tutor_action_idx_per_env = np.full((num_envs,), -1, dtype=np.int64)
            tutee_action_idx_per_env = np.full((num_envs,), -1, dtype=np.int64)
            ll_action_strs = ["no_help"] * num_envs

            # Tutor groups by topic
            for t in range(num_topics):
                idx = np.where((~hl_modes) & (hl_topics == t) & (~done))[0]
                if idx.size == 0:
                    continue

                topic_col = torch.full((idx.size, 1), float(t), device=device)
                ll_obs = torch.cat([obs_t[idx], topic_col], dim=1)

                ll_actions = tutors[t].select_action_batch(ll_obs).detach().cpu().numpy().astype(np.int64)
                tutor_action_idx_per_env[idx] = ll_actions

                for k, env_i in enumerate(idx.tolist()):
                    a_str = tutor_action_names[int(ll_actions[k])]
                    ll_action_strs[env_i] = a_str
                    tutor_action_counts[a_str] += 1

                topic_counts[t] += idx.size

            # Tutee group
            if use_tutee and tutee is not None:
                idx = np.where((hl_modes) & (~done))[0]
                if idx.size > 0:
                    topic_col = torch.from_numpy(hl_topics[idx]).to(device=device).float().unsqueeze(1)
                    ll_obs = torch.cat([obs_t[idx], topic_col], dim=1)

                    ll_actions = tutee.select_action_batch(ll_obs).detach().cpu().numpy().astype(np.int64)
                    tutee_action_idx_per_env[idx] = ll_actions

                    for k, env_i in enumerate(idx.tolist()):
                        a_str = tutee_action_names[int(ll_actions[k])]
                        ll_action_strs[env_i] = a_str
                        tutee_action_counts[a_str] += 1

                    for t in hl_topics[idx]:
                        topic_counts[int(t)] += 1

            # Step envs
            next_obs, rewards, done2 = env.step(hl_modes, hl_topics, ll_action_strs, done)

            # Training updates (batched)
            active_idx = np.where(~done)[0]
            if active_idx.size > 0:
                actions_cpu = hl_actions.detach().cpu()
                hl.update_batch(
                    obs=torch.from_numpy(obs[active_idx]),
                    actions=actions_cpu[active_idx],
                    rewards=torch.from_numpy(rewards[active_idx]),
                    next_obs=torch.from_numpy(next_obs[active_idx]),
                    dones=torch.from_numpy(done2[active_idx].astype(np.float32)),
                )

                # Tutor updates by topic
                for t in range(num_topics):
                    idx = np.where((~hl_modes) & (hl_topics == t) & (~done))[0]
                    if idx.size == 0:
                        continue
                    ll_obs = np.concatenate([obs[idx], np.full((idx.size, 1), float(t), dtype=np.float32)], axis=1)
                    ll_next = np.concatenate([next_obs[idx], np.full((idx.size, 1), float(t), dtype=np.float32)], axis=1)
                    a_idx = tutor_action_idx_per_env[idx]
                    tutors[t].update_batch(
                        obs=torch.from_numpy(ll_obs),
                        actions=torch.from_numpy(a_idx),
                        rewards=torch.from_numpy(rewards[idx]),
                        next_obs=torch.from_numpy(ll_next),
                        dones=torch.from_numpy(done2[idx].astype(np.float32)),
                    )

                # Tutee updates
                if use_tutee and tutee is not None:
                    idx = np.where((hl_modes) & (~done))[0]
                    if idx.size > 0:
                        ll_obs = np.concatenate([obs[idx], hl_topics[idx].astype(np.float32).reshape(-1, 1)], axis=1)
                        ll_next = np.concatenate([next_obs[idx], hl_topics[idx].astype(np.float32).reshape(-1, 1)], axis=1)
                        a_idx = tutee_action_idx_per_env[idx]
                        tutee.update_batch(
                            obs=torch.from_numpy(ll_obs),
                            actions=torch.from_numpy(a_idx),
                            rewards=torch.from_numpy(rewards[idx]),
                            next_obs=torch.from_numpy(ll_next),
                            dones=torch.from_numpy(done2[idx].astype(np.float32)),
                        )

            total_reward[~done] += rewards[~done]
            obs = next_obs
            done = done2
            steps += 1

        if (batch_idx % log_every_batches) == 0 or batch_idx == 1:
            mean_reward = float(np.mean(total_reward))
            mastery = np.array([sum(e.state.mastery_learner) / e.num_topics for e in env.envs], dtype=np.float32)
            mean_mastery = float(mastery.mean())

            total_topic = int(topic_counts.sum()) or 1
            topic_freq = topic_counts / total_topic

            print(f"[Batch {batch_idx:4d}/{total_batches}] eps={eps:.3f}")
            print(f"  Mean reward (per learner): {mean_reward:.4f}")
            print(f"  Mean steps (vector horizon): {int(steps)}")
            print(f"  Mean learner mastery: {mean_mastery:.3f}")
            print("  Topic choice frequencies:")
            for t in range(num_topics):
                print(f"    - Topic {t}: {topic_freq[t] * 100:5.1f}%")

            total_tutor = sum(tutor_action_counts.values()) or 1
            print("  Tutor action frequencies:")
            for a in tutor_action_names:
                print(f"    - {a:20s}: {tutor_action_counts[a] / total_tutor * 100:5.1f}%")

            if use_tutee and tutee is not None:
                total_tutee = sum(tutee_action_counts.values()) or 1
                print("  Tutee action frequencies:")
                for a in tutee_action_names:
                    print(f"    - {a:20s}: {tutee_action_counts[a] / total_tutee * 100:5.1f}%")
            print()

    print("Training finished.")


if __name__ == "__main__":
    train_vectorized()
