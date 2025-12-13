# agents/low_level_agents.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Hashable, List
import random
import math

StateType = Tuple[Hashable, ...]


@dataclass
class LowLevelAgentConfig:
    num_topics: int
    num_buckets = 5
    alpha: float = 0.001
    gamma: float = 0.9
    epsilon: float = 0.1
    state_rounding: int = 2


class TabularLowLevelAgent:
    """
    Generic tabular Q-learning agent for low-level decisions.
    Concrete subclasses only need to define `actions` (list of str).
    """

    def __init__(self, config: LowLevelAgentConfig, actions: List[str]):
        self.cfg = config
        self.actions = actions
        self.q_table: Dict[Tuple[StateType, int], float] = {}

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def get_action_meanings(self) -> List[str]:
        return self.actions

    def select_action(self, obs: List[float]) -> int:
        state = self._encode_state(obs)
        if random.random() < self.cfg.epsilon:
            return random.randrange(self.num_actions)

        q_vals = [self.q_table.get((state, a), 0.0) for a in range(self.num_actions)]
        max_q = max(q_vals)
        best_actions = [a for a, q in enumerate(q_vals) if math.isclose(q, max_q)]
        return random.choice(best_actions)

    def set_epsilon(self, epsilon: float) -> None:
        self.cfg.epsilon = max(0.0, float(epsilon))

    def update(
            self,
            obs: List[float],
            action: int,
            reward: float,
            next_obs: List[float],
            done: bool,
    ) -> None:
        state = self._encode_state(obs)
        next_state = self._encode_state(next_obs)

        key = (state, action)
        old_q = self.q_table.get(key, 0.0)

        if done:
            target = reward
        else:
            next_qs = [self.q_table.get((next_state, a), 0.0) for a in range(self.num_actions)]
            target = reward + self.cfg.gamma * max(next_qs, default=0.0)

        new_q = old_q + self.cfg.alpha * (target - old_q)
        self.q_table[key] = new_q

    # --------- internal helpers ---------

    def _encode_state(self, obs: List[float]) -> StateType:
        # r = self.cfg.state_rounding
        # return tuple(round(x, r) for x in obs)
        """
        Expect obs to be: [full_high_level_obs..., topic_id].
        We compress this to (topic_id, mastery_bucket) for the learner.
        """
        topic_id = int(round(obs[-1]))  # last element is topic_id
        # num_topics = 8  # TODO pass via config
        num_topics = self.cfg.num_topics
        bucket_count = self.cfg.num_buckets
        learner_mastery = obs[0:num_topics]  # first num_topics: learner mastery

        m = learner_mastery[topic_id]
        bucket = min(bucket_count - 1, int(m * bucket_count))
        # if m < 0.33:
        #     bucket = 0
        # elif m < 0.66:
        #     bucket = 1
        # else:
        #     bucket = 2

        return topic_id, bucket


# ---------- TUTOR low-level agents ----------

def build_tutor_actions() -> List[str]:
    """
    Discrete assistance types given by the Tutor to the learner.
    """
    return [
        "hint",  # small nudge
        "worked_example",  # show full solution
        "reflection_question",  # ask the learner to think/explain
        "no_help",  # let learner struggle / practice
    ]


class TutorLowLevelAgent(TabularLowLevelAgent):
    """
    One instance per topic
    """

    def __init__(self, config: LowLevelAgentConfig):
        super().__init__(config, actions=build_tutor_actions())


# ---------- TUTEE low-level agent ----------

def build_tutee_actions() -> List[str]:
    """
    Discrete requests made by the Tutee to the learner.
    """
    return [
        "ask_explanation",  # "Can you explain this to me?"
        "ask_worked_example",  # "Can you show me how to solve this?"
        "ask_summary",  # "Can you summarize this topic?"
        "show_mistake_and_ask_fix",  # tutee presents possibly-wrong solution; learner must correct
    ]


class TuteeLowLevelAgent(TabularLowLevelAgent):
    """
    Single instance for the whole system.
    Topic information should be part of the state passed to select_action().
    """

    def __init__(self, config: LowLevelAgentConfig):
        super().__init__(config, actions=build_tutee_actions())
