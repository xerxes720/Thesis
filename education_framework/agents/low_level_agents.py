import random
from collections import deque, defaultdict
from typing import Any, List, Tuple

State = Any
Action = int
Transition = Tuple[State, Action, float, State, bool]


class ReplayBuffer:
    """Simple FIFO replay buffer."""
    def __init__(self, capacity: int):
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


class LowLevelAgent:
    """
    Simplified low-level agent (tabular Q-learning) with its own replay buffer.
    This corresponds to A^n and Ψ^n in Algorithm 1 of the paper, but we use
    a Q-table instead of a neural network.
    """

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        lr: float = 0.1,
        buffer_capacity: int = 10000,
    ):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.memory = ReplayBuffer(buffer_capacity)

        # Q-table: maps (state, action) -> value
        self.q_table = defaultdict(float)

    # ---------- interaction ----------

    def select_action(self, state: State) -> Action:
        """ε-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def store_transition(self, s: State, a: Action, r: float, s_next: State, done: bool):
        self.memory.add((s, a, r, s_next, done))

    # ---------- learning ----------

    def _update_from_sample(
        self,
        s: State,
        a: Action,
        r: float,
        s_next: State,
        done: bool,
        weight: float,
    ):
        """Single Q-learning update, scaled by 'weight' (for shared samples)."""
        old_q = self.q_table[(s, a)]

        if done:
            target = r
        else:
            next_qs = [self.q_table[(s_next, a2)] for a2 in range(self.n_actions)]
            target = r + self.gamma * max(next_qs)

        td_error = target - old_q
        self.q_table[(s, a)] = old_q + self.lr * weight * td_error

    def train_with_batch(
        self,
        own_batch: List[Transition],
        shared_batches: List[Tuple[Transition, float]],
        own_weight: float = 1.0,
    ):
        """
        Apply Q-learning updates on own_batch and shared_batches.
        own_batch samples have weight 'own_weight' (typically 1.0).
        shared_batches items are (transition, weight).
        """
        # Own experience
        for (s, a, r, s_next, done) in own_batch:
            self._update_from_sample(s, a, r, s_next, done, own_weight)

        # Shared experience from other agents
        for (transition, w) in shared_batches:
            s, a, r, s_next, done = transition
            self._update_from_sample(s, a, r, s_next, done, w)


# -----------------------------------------------------------------------------
# Experience sharing trainer (simplified Algorithm 1)
# -----------------------------------------------------------------------------

def train_low_level_agent_with_experience_sharing(
    agents: List[LowLevelAgent],
    i: int,
    batch_size: int,
    shared_fraction: float = 0.5,
    shared_weight: float = 0.3,
):
    """
    Simplified version of Algorithm 1 in the paper:

    - Sample a batch from agent i's own replay buffer (weight = 1.0).
    - Sample smaller batches from other agents' buffers (weight = shared_weight).
    - Combine all samples into one batch and update agent i's Q-table.

    This replaces CKA-based similarity with a constant shared_weight
    (a surrogate for 'usefulness' of other agents' experience).
    """

    agent_i = agents[i]

    # If the agent doesn't have enough own experience yet, skip training
    if len(agent_i.memory) == 0:
        return

    # 1) Own batch B^i  (Algorithm 1 lines 10 & 12 idea)
    own_batch = agent_i.memory.sample(batch_size)

    # 2) Shared batches from other agents (Algorithm 1 lines 5–8 & 11–12)
    shared_batches: List[Tuple[Transition, float]] = []

    # size for each other agent
    per_agent_shared = max(1, int(batch_size * shared_fraction / max(1, len(agents) - 1)))

    for j, agent_j in enumerate(agents):
        if j == i or len(agent_j.memory) == 0:
            continue

        Bj = agent_j.memory.sample(per_agent_shared)
        for transition in Bj:
            # we attach shared_weight instead of computing CKA
            shared_batches.append((transition, shared_weight))

    # 3) Train agent i on combined batch (Algorithm 1 line 13)
    agent_i.train_with_batch(own_batch, shared_batches, own_weight=1.0)
