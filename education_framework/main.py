# main.py

from agents.high_level_agent import HighLevelAgent, HighLevelAgentConfig
from agents.low_level_agents import (
    TutorLowLevelAgent,
    TuteeLowLevelAgent,
    LowLevelAgentConfig,
)
from environment.learner_model import LearnerModel


def create_agents(num_topics: int, use_tutee: bool = True):
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
    Runs one episode and optionally trains agents on the fly.
    Returns total reward and number of steps.
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        # --- High-level decision ---
        hl_action_idx = high_level_agent.select_action(obs)
        mode, topic_id = high_level_agent.decode_action(hl_action_idx)

        # --- Low-level decision & environment step ---
        if mode == "tutor":
            # low-level tutor for this topic
            tutor_agent = tutor_agents[topic_id]

            # (optional) include topic_id as extra feature for low-level
            tutor_obs = obs + [float(topic_id)]

            ll_action_idx = tutor_agent.select_action(tutor_obs)
            ll_action_str = tutor_agent.get_action_meanings()[ll_action_idx]

            next_obs, reward, done, _ = env.step_tutor(topic_id, ll_action_str)

            if train:
                # Update high-level agent
                high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
                # Update low-level tutor agent
                tutor_agent.update(tutor_obs, ll_action_idx, reward, next_obs + [float(topic_id)], done)

        elif mode == "tutee":
            if tutee_agent is None:
                # Fallback: if tutee disabled, just skip (shouldn't normally happen)
                next_obs, reward, done, _ = env.step_tutor(topic_id, "no_help")
                if train:
                    high_level_agent.update(obs, hl_action_idx, reward, next_obs, done)
            else:
                # (optional) include topic_id as extra feature for low-level tutee agent
                tutee_obs = obs + [float(topic_id)]

                ll_action_idx = tutee_agent.select_action(tutee_obs)
                ll_action_str = tutee_agent.get_action_meanings()[ll_action_idx]

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
    num_topics = 3
    use_tutee = True

    # Environment
    env = LearnerModel(num_topics=num_topics)

    # Agents
    high_level_agent, tutor_agents, tutee_agent = create_agents(
        num_topics=num_topics,
        use_tutee=use_tutee,
    )

    num_episodes = 500

    for episode in range(1, num_episodes + 1):
        total_reward, steps = run_episode(
            env,
            high_level_agent,
            tutor_agents,
            tutee_agent,
            train=True,
        )

        if episode % 20 == 0:
            avg_mastery = sum(env.state.mastery_learner) / env.num_topics
            print(
                f"Episode {episode:4d} | "
                f"total_reward={total_reward: .4f} | "
                f"steps={steps:3d} | "
                f"avg_mastery={avg_mastery: .3f}"
            )


if __name__ == "__main__":
    main()
