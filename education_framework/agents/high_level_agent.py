"""
High-Level Agent
----------------
Simplified high-level controller for the hierarchical tutoring system.

This agent chooses among:
- Subtasks (e.g., topic 0, 1, 2, ...)
- Optionally: "invoke tutee" as a special high-level action

It uses a tabular Q-learning approach with an epsilon-greedy policy.
The state is assumed to be a small, hashable representation of the
learner/environment (e.g., a tuple of mastery/motivation/error/retention
and maybe a subtask progress flag).

The decoding of high-level actions is:
    action_id in [0, n_subtasks - 1]  -> select subtask `action_id`
    if include_tutee:
        action_id == n_subtasks      -> invoke tutee interaction
"""

import random
from collections import defaultdict, deque
from typing import Any, Tuple, List


State = Any
Action = int
Transition = Tuple[State, Action, float, State, bool]


class ReplayBuffer:
    """Minimal replay buffer for high-level experiences."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        if len(self.buffer) == 0:
            return []
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class HighLevelAgent:
    """
    Tabular Q-learning high-level agent.

    Parameters
    ----------
    n_subtasks : int
        Number of subtasks (topics) the learner can work on.
    include_tutee : bool
        If True, an extra high-level action "invoke tutee" is added.
    epsilon : float
        Exploration probability for epsilon-greedy policy.
    gamma : float
        Discount factor.
    lr : float
        Learning rate for Q updates.
    buffer_capacity : int
        Capacity of the replay buffer.
    """

    def __init__(
        self,
        n_subtasks: int,
        include_tutee: bool = True,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        lr: float = 0.1,
        buffer_capacity: int = 10000,
    ):
        self.n_subtasks = n_subtasks
        self.include_tutee = include_tutee
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

        # number of available high-level actions
        self.n_actions = n_subtasks + (1 if include_tutee else 0)

        # Q-table: maps (state, action) -> value
        self.q_table = defaultdict(float)

        # replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: State) -> Action:
        """
        Epsilon-greedy selection over high-level actions.

        Returns
        -------
        action_id : int
            Index in [0, n_actions - 1]:
              - 0..n_subtasks-1: choose that subtask
              - n_subtasks:     invoke tutee (if include_tutee=True)
        """
        # exploration
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        # exploitation: pick argmax_a Q(s, a)
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def decode_action(self, action_id: int):
        """
        Convert integer action_id to a semantic decision.

        Returns
        -------
        decision_type : str
            "subtask" or "tutee"
        value : int or None
            If "subtask": subtask index (0..n_subtasks-1)
            If "tutee": None
        """
        if action_id < self.n_subtasks:
            return "subtask", action_id
        elif self.include_tutee and action_id == self.n_subtasks:
            return "tutee", None
        else:
            raise ValueError(f"Invalid action_id {action_id} for configuration.")

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ):
        """
        Store a high-level transition in replay buffer.
        """
        self.memory.add((state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def train(self, batch_size: int = 32):
        """
        Sample a batch from replay buffer and perform Q-learning updates.
        If there's not enough data, this function is a no-op.
        """
        if len(self.memory) == 0:
            return

        batch = self.memory.sample(batch_size)

        for (s, a, r, s_next, done) in batch:
            old_q = self.q_table[(s, a)]

            if done:
                target = r
            else:
                # max_a' Q(s_next, a')
                next_qs = [self.q_table[(s_next, a2)] for a2 in range(self.n_actions)]
                target = r + self.gamma * max(next_qs)

            td_error = target - old_q
            self.q_table[(s, a)] = old_q + self.lr * td_error

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_epsilon(self, epsilon: float):
        """Update exploration rate (for annealing)."""
        self.epsilon = epsilon

    def __repr__(self):
        return (
            f"HighLevelAgent(n_subtasks={self.n_subtasks}, "
            f"include_tutee={self.include_tutee}, epsilon={self.epsilon})"
        )
