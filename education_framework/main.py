# main.py

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
    # High-level agent (decides topic + mode: tutor vs tutee)
    hl_cfg = HighLevelAgentConfig(num_topics=num_topics, use_tutee=use_tutee)
    high_level_agent = HighLevelAgent(hl_cfg)

    # Low-level tutor agents (one per topic for now)
    ll_cfg = LowLevelAgentConfig()
    tutor_agents = [TutorLowLevelAgent(ll_cfg) for _ in range(num_topics)]

    # Single tutee low-level agent
    tutee_agent = TuteeLowLevelAgent(ll_cfg) if use_tutee else None

    return high_level_agent, tutor_agents, tutee_agent


def run_episode(env, high_level_agent, tutor_agents, tutee_agent, train: bool = True):
    """
    Run one full learning episode.

    Returns:
      - total_reward: sum of rewards over the episode
      - steps: number of actions taken (tutor or tutee interactions)
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        # === High-level decision: choose (mode, topic) ===
        hl_action_idx = high_level_agent.select_action(obs)
        mode, topic_id = high_level_agent.decode_action(hl_action_idx)

        # === Low-level decision & environment step ===
        if mode == "tutor":
            # low-level tutor for this topic
            tutor_agent = tutor_agents[topic_id]

            # include topic_id as extra feature for low-level agent (optional but useful)
            tutor_obs = obs + [float(topic_id)]

            ll_action_idx = tutor_agent.select_action(tutor_obs)
            ll_action_str = tutor_agent.get_action_meanings()[ll_action_idx]

            # apply tutor action in environment
            next_obs, reward, done, _ = env.step_tutor(topic_id, ll_action_str)

            if train:
                # Update high-level agent
                high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
                # Update low-level tutor agent
                tutor_agent.update(tutor_obs, ll_action_idx, reward, next_obs + [float(topic_id)], done)

        elif mode == "tutee":
            if tutee_agent is None:
                # Fallback: if tutee is disabled, just do a no_help tutor step
                next_obs, reward, done, _ = env.step_tutor(topic_id, "no_help")
                if train:
                    high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
            else:
                # include topic_id as extra feature for low-level tutee agent
                tutee_obs = obs + [float(topic_id)]

                ll_action_idx = tutee_agent.select_action(tutee_obs)
                ll_action_str = tutee_agent.get_action_meanings()[ll_action_idx]

                # apply tutee interaction in environment (learner teaches tutee)
                next_obs, reward, done, _ = env.step_tutee(topic_id, ll_action_str)

                if train:
                    # Update high-level agent
                    high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
                    # Update tutee low-level agent
                    tutee_agent.update(tutee_obs, ll_action_idx, reward, next_obs + [float(topic_id)], done)

        else:
            # Safety fallback: treat as tutor no_help
            next_obs, reward, done, _ = env.step_tutor(topic_id, "no_help")
            if train:
                high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)

        total_reward += reward
        steps += 1
        obs = next_obs

    return total_reward, steps


def main():
    # --- Configuration ---
    num_topics = 3
    use_tutee = False          # True = learner sometimes *teaches* the tutee
    num_episodes = 1000

    # --- Environment and agents ---
    env = LearnerModel(num_topics=num_topics)
    high_level_agent, tutor_agents, tutee_agent = create_agents(
        num_topics=num_topics,
        use_tutee=use_tutee,
    )

    print("=== Training hierarchical RL tutor ===")
    print(f"- Number of topics: {num_topics}")
    print(f"- Tutee enabled:   {use_tutee}")
    print(f"- Episodes:        {num_episodes}\n")

    window = 50
    rewards_window = []
    mastery_window = []
    tutee_window = []

    for episode in range(1, num_episodes + 1):
        total_reward, steps = run_episode(
            env,
            high_level_agent,
            tutor_agents,
            tutee_agent,
            train=True,
        )

        avg_mastery_learner = sum(env.state.mastery_learner) / env.num_topics
        avg_mastery_tutee = sum(env.state.mastery_tutee) / env.num_topics

        rewards_window.append(total_reward)
        mastery_window.append(avg_mastery_learner)
        tutee_window.append(avg_mastery_tutee)

        if episode % window == 0:
            print(
                f"[Episode {episode:4d}] "
                f"Mean reward (last {window}): {sum(rewards_window) / window: .4f} | "
                f"Mean learner mastery:       {sum(mastery_window) / window: .3f} | "
                f"Mean tutee mastery:         {sum(tutee_window) / window: .3f}"
            )
            rewards_window.clear()
            mastery_window.clear()
            tutee_window.clear()
        # We log every 20 episodes to see the learning trend
    #     if episode % 20 == 0:
    #         avg_mastery_learner = sum(env.state.mastery_learner) / env.num_topics
    #         avg_mastery_tutee = sum(env.state.mastery_tutee) / env.num_topics
    #         print(
    #             f"[Episode {episode:4d}] "
    #             f"Total reward (sum over steps): {total_reward: .4f} | "
    #             f"Steps in episode: {steps:3d}\n"
    #             f"   -> Avg learner mastery: {avg_mastery_learner: .3f} "
    #             f"(0 = no knowledge, 1 = mastered)\n"
    #             f"   -> Avg tutee mastery:   {avg_mastery_tutee: .3f} "
    #             f"(tutee's knowledge level)\n"
    #         )
    #
    # print("Training finished.\n")
    # print("Interpretation of metrics:")
    # print("- Total reward: how well the tutoring/teaching decisions improved")
    # print("  learner mastery, motivation, error rate, and retention in that episode.")
    # print("- Avg learner mastery: overall learning progress of the human learner model.")
    # print("- Avg tutee mastery: how much the apprentice NPC has learned from being taught.")


if __name__ == "__main__":
    main()
