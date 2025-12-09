# environment/learner_model.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import random
import math


@dataclass
class LearnerTuteeState:
    """
    Holds the internal state of the human learner + tutee.

    - mastery_learner[i]: learner's mastery in topic i ∈ [0,1]
    - mastery_tutee[i]: tutee's mastery in topic i ∈ [0,1]
    - motivation: learner's motivation ∈ [0,1]
    - error_rate: approximate probability of making mistakes ∈ [0,1]
    - retention: how stable the learned knowledge is ∈ [0,1]
    """
    num_topics: int
    mastery_learner: List[float] = field(init=False)
    mastery_tutee: List[float] = field(init=False)
    motivation: float = 0.7
    error_rate: float = 0.3
    retention: float = 0.5
    step_count: int = 0

    def __post_init__(self):
        # initialize with random or low mastery
        self.mastery_learner = [random.uniform(0.1, 0.4) for _ in range(self.num_topics)]
        self.mastery_tutee = [random.uniform(0.0, 0.2) for _ in range(self.num_topics)]

    def clone(self) -> "LearnerTuteeState":
        copy = LearnerTuteeState(num_topics=self.num_topics)
        copy.mastery_learner = self.mastery_learner[:]
        copy.mastery_tutee = self.mastery_tutee[:]
        copy.motivation = self.motivation
        copy.error_rate = self.error_rate
        copy.retention = self.retention
        copy.step_count = self.step_count
        return copy


class LearnerModel:
    """
    Abstract simulated environment for:
      - a human learner
      - an apprentice/tutee agent

    This does NOT implement the RL loop; it just:
      - stores state,
      - applies tutor and tutee actions,
      - computes rewards.

    High-level + low-level agents will use this class inside a training loop.
    """

    def __init__(self, num_topics: int):
        self.num_topics = num_topics
        self.state = LearnerTuteeState(num_topics=num_topics)

        # reward weights (you can tweak these)
        self.w_mastery = 1.0
        self.w_motivation = 0.3
        self.w_error = 0.5
        self.w_retention = 0.2

        # threshold to consider an episode "done"
        self.mastery_target = 0.95
        self.max_steps = 80
        self.prereqs = {
            1: [0],  # to learn topic 1 well, you need topic 0
            2: [1],  # to learn topic 2, you need topic 1
        }

    # --------------- core API ----------------

    def reset(self) -> List[float]:
        self.state = LearnerTuteeState(num_topics=self.num_topics)
        return self.get_observation()

    def get_observation(self) -> List[float]:
        """
        Flatten state into a numeric vector for RL agents.

        Current design:
          [ mastery_learner..., mastery_tutee..., motivation, error_rate, retention ]
        """
        obs = []
        obs.extend(self.state.mastery_learner)
        obs.extend(self.state.mastery_tutee)
        obs.append(self.state.motivation)
        obs.append(self.state.error_rate)
        obs.append(self.state.retention)
        return obs

    def step_tutor(self, topic_id: int, tutor_action: str) -> Tuple[List[float], float, bool, Dict]:
        """
        Apply a Tutor low-level action on a given topic.

        Returns: next_obs, reward, done, info
        """
        prev_state = self.state.clone()
        self._apply_tutor_action(topic_id, tutor_action)
        self.state.step_count += 1

        reward = self._compute_reward(prev_state, self.state)
        # # high cost
        # if tutor_action == "worked_example":
        #     reward -= 1
        done = self._check_done()
        return self.get_observation(), reward, done, {}

    def step_tutee(self, topic_id: int, tutee_action: str) -> Tuple[List[float], float, bool, Dict]:
        """
        Apply a Tutee low-level action on a given topic.

        Returns: next_obs, reward, done, info
        """
        prev_state = self.state.clone()
        self._apply_tutee_action(topic_id, tutee_action)
        self.state.step_count += 1

        reward = self._compute_reward(prev_state, self.state)
        done = self._check_done()
        return self.get_observation(), reward, done, {}

    # --------------- internal dynamics ----------------

    def _prereq_factor(self, topic_id: int) -> float:
        """Return how 'ready' the learner is for this topic based on prereqs."""
        if topic_id not in self.prereqs:
            return 1.0  # no prereqs → full learning rate

        prereq_ids = self.prereqs[topic_id]
        if not prereq_ids:
            return 1.0

        # e.g. average mastery over prerequisites
        avg_prereq_mastery = sum(self.state.mastery_learner[i] for i in prereq_ids) / len(prereq_ids)

        # map [0,1] → [0.2, 1.0] so it's never completely zero
        return 0.2 + 0.8 * avg_prereq_mastery

    def _apply_tutor_action(self, topic_id: int, action: str) -> None:
        """
        Simplified tutor effects on the learner.

        You can refine these formulas later if needed.
        """
        M = self.state.mastery_learner[topic_id]
        m = self.state.motivation
        e = self.state.error_rate
        r = self.state.retention

        # base learning rate depends on motivation and retention
        base_gain = 0.05 + 0.1 * m + 0.05 * r

        if action == "hint":
            delta_M = base_gain * 0.6 * (1 - M)
            delta_m = 0.01
            delta_e = -0.02
        elif action == "worked_example":
            delta_M = base_gain * 1.0 * (1 - M)
            delta_m = 0.0
            delta_e = -0.03
        elif action == "reflection_question":
            # harder, more gain if motivation is high; can hurt if motivation low
            if m > 0.5:
                delta_M = base_gain * 1.1 * (1 - M)
                delta_m = 0.02
            else:
                delta_M = base_gain * 0.5 * (1 - M)
                delta_m = -0.02
            delta_e = -0.01
        elif action == "no_help":
            # pure practice: small gain, error may increase slightly
            delta_M = base_gain * 0.2 * (1 - M)
            delta_m = 0.0
            delta_e = 0.05
        else:
            # unknown action: no change
            delta_M = 0.0
            delta_m = 0.0
            delta_e = 0.0

        factor = self._prereq_factor(topic_id)
        delta_M *= factor
        # apply with noise
        noise = random.gauss(0.0, 0.01)
        self.state.mastery_learner[topic_id] = _clip01(M + delta_M + noise)
        self.state.motivation = _clip01(m + delta_m)
        self.state.error_rate = _clip01(e + delta_e)
        # retention grows slowly as mastery increases
        self.state.retention = _clip01(r + 0.02 * delta_M)

    def _apply_tutee_action(self, topic_id: int, action: str) -> None:
        """
        Tutee request + learner teaching attempt.

        Implements the protégé effect:
          - learner gains extra mastery when successfully teaching
          - tutee's own mastery is updated
        """
        ML = self.state.mastery_learner[topic_id]
        MT = self.state.mastery_tutee[topic_id]
        m = self.state.motivation
        e = self.state.error_rate
        r = self.state.retention

        # probability that learner gives a good explanation depends on learner mastery + motivation
        p_success = _clip01(0.2 + 0.6 * ML + 0.2 * m)
        p_partial = _clip01(0.1 + 0.3 * ML)
        # re-normalize
        total = p_success + p_partial
        if total > 1.0:
            p_success /= total
            p_partial /= total
        p_fail = 1.0 - (p_success + p_partial)

        outcome = _sample_outcome(p_success, p_partial, p_fail)

        # base teaching gain
        base_gain = 0.04 + 0.08 * m + 0.04 * r
        protege_bonus = 0.03  # extra gain for learner when teaching succeeds

        if action == "ask_explanation":
            learner_mult = 1.2
            tutee_mult = 1.0
        elif action == "ask_worked_example":
            learner_mult = 1.0
            tutee_mult = 1.1
        elif action == "ask_summary":
            learner_mult = 0.8
            tutee_mult = 0.8
        elif action == "show_mistake_and_ask_fix":
            learner_mult = 1.3
            tutee_mult = 1.2
        else:
            learner_mult = 1.0
            tutee_mult = 1.0

        if outcome == "success":
            delta_M_learner = base_gain * learner_mult * (1 - ML) + protege_bonus
            delta_M_tutee = base_gain * tutee_mult * (1 - MT)
            delta_m = 0.02
            delta_e = -0.02
        elif outcome == "partial":
            delta_M_learner = base_gain * 0.7 * learner_mult * (1 - ML) + protege_bonus * 0.5
            delta_M_tutee = base_gain * 0.6 * tutee_mult * (1 - MT)
            delta_m = 0.0
            delta_e = -0.01
        else:  # fail ("I don't know" / incorrect)
            delta_M_learner = -0.01  # small setback / confusion
            delta_M_tutee = base_gain * 0.2 * (1 - MT)  # tutee still learns a bit
            delta_m = -0.02
            delta_e = 0.02

        # apply with noise
        noise_L = random.gauss(0.0, 0.01)
        noise_T = random.gauss(0.0, 0.01)

        self.state.mastery_learner[topic_id] = _clip01(ML + delta_M_learner + noise_L)
        self.state.mastery_tutee[topic_id] = _clip01(MT + delta_M_tutee + noise_T)
        self.state.motivation = _clip01(m + delta_m)
        self.state.error_rate = _clip01(e + delta_e)
        self.state.retention = _clip01(r + 0.03 * max(delta_M_learner, 0.0))

    # --------------- reward & termination ----------------

    def _compute_reward(self, prev: LearnerTuteeState, cur: LearnerTuteeState) -> float:
        """
        Reward encourages:
          - increases in learner mastery (primary)
          - increases in motivation and retention
          - decreases in error_rate

        You can also experiment with including tutee mastery here if you want
        the tutor to care explicitly about the tutee.
        """
        avg_prev_mastery = sum(prev.mastery_learner) / prev.num_topics
        avg_cur_mastery = sum(cur.mastery_learner) / cur.num_topics

        delta_mastery = avg_cur_mastery - avg_prev_mastery
        delta_motivation = cur.motivation - prev.motivation
        delta_error = cur.error_rate - prev.error_rate
        delta_retention = cur.retention - prev.retention

        reward = (
            self.w_mastery * delta_mastery
            + self.w_motivation * delta_motivation
            - self.w_error * delta_error
            + self.w_retention * delta_retention
        )
        # Time cost
        reward -= 0.01

        return reward

    def _check_done(self) -> bool:
        avg_mastery = sum(self.state.mastery_learner) / self.num_topics
        if avg_mastery >= self.mastery_target:
            return True
        if self.state.step_count >= self.max_steps:
            return True
        return False


# --------------- helpers ----------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sample_outcome(p_success: float, p_partial: float, p_fail: float) -> str:
    """
    Sample one of {"success", "partial", "fail"} given probabilities.
    """
    r = random.random()
    if r < p_success:
        return "success"
    if r < p_success + p_partial:
        return "partial"
    return "fail"
