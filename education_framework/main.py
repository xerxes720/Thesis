# main.py

from collections import defaultdict

from agents.high_level_agent import HighLevelAgent, HighLevelAgentConfig
from agents.low_level_agents import (
    TutorLowLevelAgent,
    TuteeLowLevelAgent,
    LowLevelAgentConfig,
)
from environment.learner_model import LearnerModel


def create_agents(num_topics: int, use_tutee: bool = True):
    """
    Create and return:
      - one high-level agent
      - a list of low-level tutor agents (one per topic)
      - one low-level tutee agent (or None if use_tutee=False)
    """
    hl_cfg = HighLevelAgentConfig(num_topics=num_topics, use_tutee=use_tutee)
    high_level_agent = HighLevelAgent(hl_cfg)

    ll_cfg = LowLevelAgentConfig()
    tutor_agents = [TutorLowLevelAgent(ll_cfg) for _ in range(num_topics)]
    tutee_agent = TuteeLowLevelAgent(ll_cfg) if use_tutee else None

    return high_level_agent, tutor_agents, tutee_agent


def run_episode(env, high_level_agent, tutor_agents, tutee_agent, train: bool = True):
    """
    Run one full episode (a single synthetic learner).

    Returns:
      - total_reward
      - steps
      - topic_counts: how many times each topic was chosen by high-level
      - tutor_action_counts: how many times each tutor low-level action was used
      - tutee_action_counts: how many times each tutee low-level action was used
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    tutor_hl_count = 0
    tutee_hl_count = 0

    num_topics = env.num_topics
    topic_counts = [0 for _ in range(num_topics)]

    # Initialize action counters
    tutor_action_counts = defaultdict(int)
    tutee_action_counts = defaultdict(int)

    # we can get action names from any tutor agent
    tutor_action_names = tutor_agents[0].get_action_meanings()
    for a in tutor_action_names:
        tutor_action_counts[a] = 0

    if tutee_agent is not None:
        tutee_action_names = tutee_agent.get_action_meanings()
        for a in tutee_action_names:
            tutee_action_counts[a] = 0

    while not done:
        # === High-level decision ===
        hl_action_idx = high_level_agent.select_action(obs)
        mode, topic_id = high_level_agent.decode_action(hl_action_idx)
        topic_counts[topic_id] += 1

        # === Low-level + env step ===
        if mode == "tutor":
            tutor_hl_count += 1
            tutor_agent = tutor_agents[topic_id]
            tutor_obs = obs + [float(topic_id)]

            ll_action_idx = tutor_agent.select_action(tutor_obs)
            ll_action_str = tutor_agent.get_action_meanings()[ll_action_idx]
            tutor_action_counts[ll_action_str] += 1

            next_obs, reward, done, _ = env.step_tutor(topic_id, ll_action_str)

            if train:
                high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
                tutor_agent.update(
                    tutor_obs,
                    ll_action_idx,
                    reward,
                    next_obs + [float(topic_id)],
                    done,
                )

        elif mode == "tutee":
            tutee_hl_count += 1
            if tutee_agent is None:
                # if tutee disabled, fall back to no_help tutoring
                next_obs, reward, done, _ = env.step_tutor(topic_id, "no_help")
                if train:
                    high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
            else:
                tutee_obs = obs + [float(topic_id)]

                ll_action_idx = tutee_agent.select_action(tutee_obs)
                ll_action_str = tutee_agent.get_action_meanings()[ll_action_idx]
                tutee_action_counts[ll_action_str] += 1

                next_obs, reward, done, _ = env.step_tutee(topic_id, ll_action_str)

                if train:
                    high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
                    tutee_agent.update(
                        tutee_obs,
                        ll_action_idx,
                        reward,
                        next_obs + [float(topic_id)],
                        done,
                    )

        else:
            # Safety fallback
            next_obs, reward, done, _ = env.step_tutor(topic_id, "no_help")
            if train:
                high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)

        total_reward += reward
        steps += 1
        obs = next_obs

    return total_reward, steps, topic_counts, tutor_action_counts, tutee_action_counts, tutor_hl_count, tutee_hl_count


def main():
    # ---------------- config ----------------
    num_topics = 3
    use_tutee = True           # you said tutee is off for now
    num_episodes = 200
    log_window = 50

    env = LearnerModel(num_topics=num_topics)
    high_level_agent, tutor_agents, tutee_agent = create_agents(
        num_topics=num_topics,
        use_tutee=use_tutee,
    )

    print("=== Training hierarchical RL tutor ===")
    print(f"- Number of topics: {num_topics}")
    print(f"- Tutee enabled:   {use_tutee}")
    print(f"- Episodes:        {num_episodes}\n")

    # rolling stats
    window_rewards = []
    window_mastery_learner = []
    window_mastery_tutee = []
    window_steps = []
    window_topic_counts = [0 for _ in range(num_topics)]
    # get tutor action names to keep order stable
    tutor_action_names = tutor_agents[0].get_action_meanings()
    window_tutor_action_counts = {a: 0 for a in tutor_action_names}

    if tutee_agent is not None:
        tutee_action_names = tutee_agent.get_action_meanings()
        window_tutee_action_counts = {a: 0 for a in tutee_action_names}
    else:
        window_tutee_action_counts = {}

    for episode in range(1, num_episodes + 1):
        total_reward, steps, topic_counts, tutor_action_counts, tutee_action_counts, tutor_hl_count, tutee_hl_count = run_episode(
            env,
            high_level_agent,
            tutor_agents,
            tutee_agent,
            train=True,
        )

        avg_mastery_learner = sum(env.state.mastery_learner) / env.num_topics
        avg_mastery_tutee = sum(env.state.mastery_tutee) / env.num_topics

        # accumulate
        window_tutor_hl = 0
        window_tutee_hl = 0
        window_rewards.append(total_reward)
        window_mastery_learner.append(avg_mastery_learner)
        window_mastery_tutee.append(avg_mastery_tutee)
        window_steps.append(steps)
        for i in range(num_topics):
            window_topic_counts[i] += topic_counts[i]
        for a in tutor_action_names:
            window_tutor_action_counts[a] += tutor_action_counts[a]
        for a in window_tutee_action_counts:
            window_tutee_action_counts[a] += tutee_action_counts.get(a, 0)

        window_tutor_hl += tutor_hl_count
        window_tutee_hl += tutee_hl_count
        # log every log_window episodes
        if episode % log_window == 0:
            w = log_window
            mean_reward = sum(window_rewards) / w
            mean_mastery_learner = sum(window_mastery_learner) / w
            mean_mastery_tutee = sum(window_mastery_tutee) / w
            mean_steps = sum(window_steps) / w

            total_topic_choices = sum(window_topic_counts) or 1
            topic_freqs = [c / total_topic_choices for c in window_topic_counts]

            total_tutor_actions = sum(window_tutor_action_counts.values()) or 1
            tutor_action_freqs = {
                a: window_tutor_action_counts[a] / total_tutor_actions
                for a in tutor_action_names
            }

            print(f"[Episode {episode:4d}]")
            print(f"  Mean reward (last {w}):          {mean_reward: .4f}")
            print(f"  Mean steps per episode:         {mean_steps: .2f}")
            print(f"  Mean learner mastery:           {mean_mastery_learner: .3f}")
            print(f"  Mean tutee mastery:             {mean_mastery_tutee: .3f}")
            print(f"  Topic choice frequencies:")
            for i, f in enumerate(topic_freqs):
                print(f"    - Topic {i}: {f*100:5.1f}% of high-level choices")
            print(f"  Tutor action frequencies:")
            for a, f in tutor_action_freqs.items():
                print(f"    - {a:20s}: {f*100:5.1f}% of tutor actions")

            if use_tutee and window_tutee_action_counts:
                total_tutee_actions = sum(window_tutee_action_counts.values()) or 1
                print(f"  Tutee action frequencies:")
                for a, c in window_tutee_action_counts.items():
                    f = c / total_tutee_actions
                    print(f"    - {a:20s}: {f*100:5.1f}% of tutee actions")
            total_hl = window_tutor_hl + window_tutee_hl or 1
            print(f"  High-level mode frequencies (last {w} episodes):")
            print(f"    - tutor: {window_tutor_hl / total_hl * 100:5.1f}% of high-level decisions")
            print(f"    - tutee: {window_tutee_hl / total_hl * 100:5.1f}% of high-level decisions")
            print()

            # reset window stats
            window_rewards.clear()
            window_mastery_learner.clear()
            window_mastery_tutee.clear()
            window_steps.clear()
            window_topic_counts = [0 for _ in range(num_topics)]
            window_tutor_action_counts = {a: 0 for a in tutor_action_names}
            if use_tutee and tutee_agent is not None:
                window_tutee_action_counts = {a: 0 for a in tutee_action_names}

    print("Training finished.")


if __name__ == "__main__":
    main()
