# agents/high_level_agents.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Hashable, List
import random
import math


StateType = Tuple[Hashable, ...]


@dataclass
class HighLevelAgentConfig:
    num_topics: int
    use_tutee: bool = True
    alpha: float = 0.1   # learning rate
    gamma: float = 0.99  # discount factor
    epsilon: float = 0.1 # exploration rate
    state_rounding: int = 2  # decimals to round continuous state to


class HighLevelAgent:
    """
    High-level tabular Q-learning agent.

    - Chooses between:
        * PRACTICE_TOPIC_i  (send learner to tutor low-level agent on topic i)
        * TEACH_TUTEE_TOPIC_i (invoke tutee low-level agent on topic i) if use_tutee=True
    """

    def __init__(self, config: HighLevelAgentConfig):
        self.cfg = config
        self.q_table: Dict[Tuple[StateType, int], float] = {}
        self._build_action_space()

    # ----------------- public API -----------------

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def get_action_meanings(self) -> List[str]:
        """Human-readable list, useful for logging/debugging."""
        return self.actions

    def select_action(self, obs: List[float]) -> int:
        """
        Epsilon-greedy action selection given a raw observation vector.
        Returns an integer action index in [0, num_actions).
        """
        state = self._encode_state(obs)

        if random.random() < self.cfg.epsilon:
            return random.randrange(self.num_actions)

        # exploit
        q_vals = [self.q_table.get((state, a), 0.0) for a in range(self.num_actions)]
        max_q = max(q_vals)
        # break ties randomly
        best_actions = [a for a, q in enumerate(q_vals) if math.isclose(q, max_q)]
        return random.choice(best_actions)

    def update(
        self,
        obs: List[float],
        action: int,
        reward: float,
        next_obs: List[float],
        done: bool,
    ) -> None:
        """
        Standard tabular Q-learning update.
        """
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

    # ----------------- action encoding -----------------

    def decode_action(self, action: int) -> Tuple[str, int]:
        """
        Decode an action index into (mode, topic_id).

        mode ∈ {"tutor", "tutee"}
        topic_id ∈ [0, num_topics)
        """
        meaning = self.actions[action]
        # examples:
        # "tutor_topic_0", "tutee_topic_2"
        mode_str, _, topic_str = meaning.partition("_topic_")
        topic_id = int(topic_str)
        mode = "tutor" if mode_str == "tutor" else "tutee"
        return mode, topic_id

    # ----------------- internal helpers -----------------

    def _build_action_space(self) -> None:
        self.actions: List[str] = []

        # tutor actions for each topic
        for t in range(self.cfg.num_topics):
            self.actions.append(f"tutor_topic_{t}")

        # tutee actions for each topic (optional)
        if self.cfg.use_tutee:
            for t in range(self.cfg.num_topics):
                self.actions.append(f"tutee_topic_{t}")

    def _encode_state(self, obs: List[float]) -> StateType:
        """
        Convert continuous observation vector to a discrete key for the Q-table.
        Here we just round; you can replace this with a better discretization later.
        """
        r = self.cfg.state_rounding
        return tuple(round(x, r) for x in obs)
