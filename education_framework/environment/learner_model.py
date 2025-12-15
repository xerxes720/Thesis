# environment/learner_model.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
import math


# -------------------- helpers --------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


QUALITY_LEVELS = ["very_bad", "bad", "neutral", "good", "very_good"]
Q2I = {q: i for i, q in enumerate(QUALITY_LEVELS)}


def _degrade_quality(q: str, k: int = 1) -> str:
    """Move quality down by k steps (bounded)."""
    i = max(0, Q2I.get(q, 2) - k)
    return QUALITY_LEVELS[i]


def _upgrade_quality(q: str, k: int = 1) -> str:
    """Move quality up by k steps (bounded)."""
    i = min(len(QUALITY_LEVELS) - 1, Q2I.get(q, 2) + k)
    return QUALITY_LEVELS[i]


def _pct_change(cur: float, prev: float, eps: float = 1e-3) -> float:
    """
    Percentage change term similar to "average % change" concept.
    We clamp to avoid extreme blow-ups when prev is tiny.
    """
    val = (cur - prev) / (abs(prev) + eps)
    return max(-1.0, min(1.0, val))


# -------------------- state --------------------

@dataclass
class LearnerTuteeState:
    """
    Multi-variable learner state in [0,1] to mimic the paper's 'performance variables'.

    Per-topic:
      - mastery_learner[i]
      - mastery_tutee[i]
      - score[i]         (proxy for test score / knowledge performance)
      - test_speed[i]    (1 = fast, 0 = slow)  (proxy for test time)
      - segment_speed[i] (1 = fast progress, 0 = slow) (proxy for time in game segment)
      - emotion[i]       (1 = positive/engaged, 0 = negative)
      - input_quality[i] (1 = good inputs / fewer mistakes, 0 = poor)

    Global:
      - motivation
      - retention
      - accuracy          (1 = low error rate, 0 = high error rate)

    Control:
      - topic_done[i]
      - step_count
      - assist_count
    """
    num_topics: int
    mastery_learner: np.ndarray = field(init=False)
    mastery_tutee: np.ndarray = field(init=False)
    score: np.ndarray = field(init=False)
    test_speed: np.ndarray = field(init=False)
    segment_speed: np.ndarray = field(init=False)
    emotion: np.ndarray = field(init=False)
    input_quality: np.ndarray = field(init=False)
    motivation: float = 0.7
    retention: float = 0.5
    accuracy: float = 0.6
    topic_done: np.ndarray = field(init=False)  # bool array
    step_count: int = 0
    assist_count: int = 0

    def __post_init__(self):
        nt = self.num_topics
        self.mastery_learner = np.random.uniform(0.10, 0.20, nt)
        self.mastery_tutee = np.random.uniform(0.00, 0.10, nt)
        self.score = np.clip(self.mastery_learner + np.random.normal(0.0, 0.05, nt), 0, 1)
        self.test_speed = np.random.uniform(0.35, 0.55, nt)
        self.segment_speed = np.random.uniform(0.35, 0.55, nt)
        self.emotion = np.random.uniform(0.50, 0.70, nt)
        self.input_quality = np.random.uniform(0.40, 0.60, nt)
        self.topic_done = np.zeros(nt, dtype=bool)

    def clone(self) -> "LearnerTuteeState":  # Now faster with np.copy
        c = object.__new__(LearnerTuteeState)
        c.num_topics = self.num_topics
        c.mastery_learner = self.mastery_learner.copy()
        c.mastery_tutee = self.mastery_tutee.copy()
        c.score = self.score.copy()
        c.test_speed = self.test_speed.copy()
        c.segment_speed = self.segment_speed.copy()
        c.emotion = self.emotion.copy()
        c.input_quality = self.input_quality.copy()
        c.motivation = self.motivation
        c.retention = self.retention
        c.accuracy = self.accuracy
        c.topic_done = self.topic_done.copy()
        c.step_count = self.step_count
        c.assist_count = self.assist_count
        return c

# -------------------- per-topic dynamics --------------------

@dataclass
class TopicDynamics:
    """
    Topic-specific parameters to emulate different environment dynamics (φ) per topic.
    """
    difficulty: float  # >1 = harder, <1 = easier
    noise_std: float  # stochasticity of learner response
    impatience: float  # how quickly motivation/emotion drop under poor help


# -------------------- main environment --------------------

class LearnerModel:
    """
    Simulator aligned with the paper's computational experiment approach:
    - Multi-variable learner state in [0,1]
    - Action quality categorized into 5 levels
    - Category-based variable shifts (topic-specific dynamics)
    - Reward = average % change of learner variables + completion bonus
    - Diminishing returns per learner
    """

    def __init__(
            self,
            num_topics: int,
            prereqs: Optional[Dict[int, List[int]]] = None,
            seed: Optional[int] = 123,
    ):
        self.num_topics = num_topics
        self.prereqs = prereqs or {}
        self.rng = random.Random(seed)

        # termination / completion thresholds
        self.mastery_target = 0.90
        self.score_target = 0.85
        self.accuracy_target = 0.60
        self.max_steps = 500

        # reward parameters
        # self.completion_bonus = 0.5  # analogous to r_c
        # self.reward_clip = (-2.0, 3.0)  # safe default; you can widen if needed
        # self.diminishing_k = 0.02  # per-learner diminishing returns strength
        # self.reward_decay = 0.98

        self.completion_bonus = 1.5  # stronger incentive to complete a topic
        self.episode_success_bonus = 2.0  # reward when ALL topics are done (early termination)
        self.step_cost = 0.01  # per-step penalty to push fewer steps
        self.timeout_penalty = 1.0  # penalty when you hit max_steps
        self.reward_clip = (-2.0, 3.0)  # keep same range as your current setup
        self.diminishing_k = 0.08  # stronger diminishing returns than 0.02

        # build topic-specific dynamics
        self.topic_dyn: List[TopicDynamics] = []
        for t in range(num_topics):
            # deterministic per-topic randomness (so runs are reproducible across resets)
            self.rng.seed((seed or 0) * 10_000 + t * 97)
            difficulty = self.rng.uniform(0.85, 1.25)
            noise_std = self.rng.uniform(0.01, 0.03)
            impatience = self.rng.uniform(0.015, 0.05)
            self.topic_dyn.append(TopicDynamics(difficulty=difficulty, noise_std=noise_std, impatience=impatience))

        self.state = LearnerTuteeState(num_topics=num_topics)

    # --------------- core API ----------------

    def reset(self) -> List[float]:
        self.state = LearnerTuteeState(num_topics=self.num_topics)
        return self.get_observation()

    def get_observation(self) -> List[float]:
        """
        Observation vector in a fixed order.

        Order:
          - mastery_learner[0..T-1]
          - mastery_tutee[0..T-1]
          - score[0..T-1]
          - test_speed[0..T-1]
          - segment_speed[0..T-1]
          - emotion[0..T-1]
          - input_quality[0..T-1]
          - motivation, retention, accuracy
        """
        # obs = []
        # obs.extend(self.state.mastery_learner)
        # obs.extend(self.state.mastery_tutee)
        # obs.extend(self.state.score)
        # obs.extend(self.state.test_speed)
        # obs.extend(self.state.segment_speed)
        # obs.extend(self.state.emotion)
        # obs.extend(self.state.input_quality)
        # obs.append(self.state.motivation)
        # obs.append(self.state.retention)
        # obs.append(self.state.accuracy)
        # return obs
        arr = np.concatenate((
            self.state.mastery_learner, self.state.mastery_tutee, self.state.score, self.state.test_speed,
            self.state.segment_speed, self.state.emotion, self.state.input_quality,
            np.array([self.state.motivation, self.state.retention, self.state.accuracy])
        ))
        return arr  # Convert to list for agents (or change agents to accept np)

    def step_tutor(self, topic_id: int, tutor_action: str) -> Tuple[List[float], float, bool, Dict]:
        prev_snapshot = self._snapshot_for_reward(topic_id)
        self._apply_action(topic_id, mode="tutor", action=tutor_action)
        self.state.step_count += 1
        self.state.assist_count += 1

        reward = self._compute_reward(prev_snapshot, self.state, topic_id)
        done = self._check_done()
        return self.get_observation(), reward, done, {"mode": "tutor"}

    def step_tutee(self, topic_id: int, tutee_action: str) -> Tuple[List[float], float, bool, Dict]:
        prev_snapshot = self._snapshot_for_reward(topic_id)
        self._apply_action(topic_id, mode="tutee", action=tutee_action)
        self.state.step_count += 1
        self.state.assist_count += 1

        reward = self._compute_reward(prev_snapshot, self.state, topic_id)
        done = self._check_done()
        return self.get_observation(), reward, done, {"mode": "tutee"}

    def _snapshot_for_reward(self, topic_id: int) -> dict:
        """
        Capture only what is needed to compute reward efficiently.

        This avoids LearnerTuteeState.clone() per step, which is a major runtime cost.
        """
        return {
            "topic_done": bool(self.state.topic_done[topic_id]),
            "mastery_learner": self.state.mastery_learner.copy(),
            "score": self.state.score.copy(),
            "test_speed": self.state.test_speed.copy(),
            "segment_speed": self.state.segment_speed.copy(),
            "emotion": self.state.emotion.copy(),
            "input_quality": self.state.input_quality.copy(),
            "motivation": float(self.state.motivation),
            "retention": float(self.state.retention),
            "accuracy": float(self.state.accuracy),
        }

    # --------------- internal dynamics ----------------

    def _prereq_factor(self, topic_id: int) -> float:
        """
        Readiness factor based on prereqs (paper uses different dynamics; this is a defensible proxy).
        Returns [0,1]. Low prereq mastery reduces the effectiveness of help.
        """
        if topic_id not in self.prereqs:
            return 1.0
        prereq_ids = self.prereqs[topic_id] or []
        if not prereq_ids:
            return 1.0
        avg = sum(self.state.mastery_learner[i] for i in prereq_ids) / len(prereq_ids)
        return _clip01(avg)

    def _topic_complete_now(self, topic_id: int) -> bool:
        """
        Multi-criteria completion for a topic.
        """
        return (
                self.state.mastery_learner[topic_id] >= self.mastery_target
                and self.state.score[topic_id] >= self.score_target
                and self.state.accuracy >= self.accuracy_target
        )

    def _apply_action(self, topic_id: int, mode: str, action: str) -> None:
        """
        Convert the selected action into a quality category, then apply category-based variable shifts.
        """
        dyn = self.topic_dyn[topic_id]
        prereq = self._prereq_factor(topic_id)

        # Determine action "quality category" based on current learner variables
        q = self._action_quality(topic_id, mode, action)

        # If prereq readiness is low, degrade quality (harder to learn out-of-order)
        if prereq < 0.35:
            q = _degrade_quality(q, 2)
        elif prereq < 0.60:
            q = _degrade_quality(q, 1)

        # Apply category-based shifts
        self._apply_quality_shifts(topic_id, mode, q, dyn, prereq)

        # Update completion flag
        if (not self.state.topic_done[topic_id]) and self._topic_complete_now(topic_id):
            self.state.topic_done[topic_id] = True

    def _action_quality(self, topic_id: int, mode: str, action: str) -> str:
        """
        Procedural 'decision-tree-like' mapping from MULTIPLE learner variables to a 5-level
        action-quality category. This is a defensible surrogate for the paper's fitted trees.

        Uses:
          M  = mastery_learner[topic]
          S  = score[topic]
          ts = test_speed[topic]       (1 fast, 0 slow)
          ss = segment_speed[topic]    (1 fast, 0 slow)
          emo= emotion[topic]
          iq = input_quality[topic]
          mot= motivation (global)
          acc= accuracy   (global)
          ret= retention  (global)

        Returns one of: very_bad, bad, neutral, good, very_good
        """
        M = self.state.mastery_learner[topic_id]
        S = self.state.score[topic_id]
        ts = self.state.test_speed[topic_id]
        ss = self.state.segment_speed[topic_id]
        emo = self.state.emotion[topic_id]
        iq = self.state.input_quality[topic_id]
        mot = self.state.motivation
        acc = self.state.accuracy
        ret = self.state.retention

        # --- aggregate signals ---
        # "Struggle" increases when: low score, slow, poor input quality, low accuracy,
        # low emotion/motivation.
        struggle = (
                (1.0 - S) * 0.30 +
                (1.0 - ts) * 0.15 +
                (1.0 - ss) * 0.10 +
                (1.0 - iq) * 0.20 +
                (1.0 - acc) * 0.15 +
                (1.0 - emo) * 0.05 +
                (1.0 - mot) * 0.05
        )
        struggle = _clip01(struggle)

        # "Teach readiness": learner benefits from teaching when mastery/score are decent,
        # inputs/accuracy are decent, and motivation/emotion are not too low.
        teach_ready = (
                M * 0.35 +
                S * 0.25 +
                iq * 0.15 +
                acc * 0.15 +
                mot * 0.05 +
                emo * 0.05
        )
        teach_ready = _clip01(teach_ready)

        # If a topic is already done, extra instruction should be less valuable.
        topic_done = self.state.topic_done[topic_id]

        # ---------------- Tutor mode ----------------
        if mode == "tutor":
            # Completed topic: prefer practice/no_help, discourage worked examples.
            if topic_done:
                if action == "no_help":
                    return "very_good"
                if action == "reflection_question":
                    return "good" if (mot > 0.45 and emo > 0.40) else "neutral"
                if action == "hint":
                    return "neutral"
                if action == "worked_example":
                    return "bad"
                return "neutral"

            # Phase by mastery, but modulated strongly by struggle/motivation/emotion/accuracy/input quality/speed.
            if M < 0.35:
                # Early stage: worked examples are often helpful if struggle is high.
                if struggle > 0.60:
                    if action == "worked_example":
                        return "very_good"
                    if action == "hint":
                        return "good"
                    if action == "reflection_question":
                        return "neutral" if mot > 0.55 else "bad"
                    if action == "no_help":
                        return "very_bad"
                else:
                    # Learner not in severe struggle: hint + example are both good.
                    if action == "worked_example":
                        return "good"
                    if action == "hint":
                        return "very_good"
                    if action == "reflection_question":
                        return "neutral" if (mot > 0.55 and emo > 0.45) else "bad"
                    if action == "no_help":
                        return "bad"
                return "neutral"

            if 0.35 <= M < 0.70:
                # Mid stage: prefer scaffolding (hint/reflection) unless struggle is high.
                if struggle > 0.65:
                    if action == "worked_example":
                        return "very_good"
                    if action == "hint":
                        return "good"
                    if action == "reflection_question":
                        return "neutral" if (mot > 0.55 and acc > 0.55) else "bad"
                    if action == "no_help":
                        return "very_bad" if (mot < 0.5 or emo < 0.45) else "bad"
                elif struggle > 0.35:
                    if action == "hint":
                        return "very_good"
                    if action == "reflection_question":
                        # reflection works if learner has enough affect + accuracy
                        return "good" if (mot > 0.55 and emo > 0.45 and acc > 0.55) else "neutral"
                    if action == "worked_example":
                        return "good" if S < 0.60 else "neutral"
                    if action == "no_help":
                        return "neutral" if mot > 0.55 else "bad"
                else:
                    # Low struggle: push toward autonomy and reflection, minimize worked examples
                    if action == "reflection_question":
                        return "very_good" if (mot > 0.55 and emo > 0.45) else "good"
                    if action == "no_help":
                        return "good" if (mot > 0.50 and acc > 0.55) else "neutral"
                    if action == "hint":
                        return "neutral"
                    if action == "worked_example":
                        return "bad" if S > 0.75 else "neutral"
                return "neutral"

            # M >= 0.70
            # Advanced: reflection/practice is best; worked example is usually wasteful; hint only if motivation/emotion low.
            if struggle > 0.55:
                # Even advanced learners can struggle (e.g., low accuracy/inputs). Use hints/targeted reflection.
                if action == "hint":
                    return "very_good"
                if action == "reflection_question":
                    return "good" if (mot > 0.50 and emo > 0.45) else "neutral"
                if action == "no_help":
                    return "neutral" if mot > 0.55 else "bad"
                if action == "worked_example":
                    return "neutral" if S < 0.70 else "bad"
            else:
                if action == "reflection_question":
                    return "very_good" if (mot > 0.50 and emo > 0.45) else "good"
                if action == "no_help":
                    return "very_good" if (mot > 0.55 and acc > 0.60) else "good"
                if action == "hint":
                    return "good" if (mot < 0.45 or emo < 0.40) else "neutral"
                if action == "worked_example":
                    return "bad"
            return "neutral"

        # ---------------- Tutee mode ----------------
        if mode == "tutee":
            # If topic already done, teaching is generally good for consolidation if motivation not too low.
            if topic_done and mot > 0.40:
                if action == "show_mistake_and_ask_fix":
                    return "very_good" if teach_ready > 0.70 else "good"
                if action == "ask_explanation":
                    return "good"
                if action == "ask_summary":
                    return "good" if ret > 0.45 else "neutral"
                if action == "ask_worked_example":
                    return "neutral"
                return "neutral"

            # Low teach readiness: forcing explanation/fix is risky; worked-example requests are safer.
            if teach_ready < 0.45:
                if action == "ask_worked_example":
                    return "good" if struggle > 0.55 else "neutral"
                if action == "ask_explanation":
                    return "neutral" if mot > 0.55 else "bad"
                if action == "ask_summary":
                    return "neutral" if emo > 0.45 else "bad"
                if action == "show_mistake_and_ask_fix":
                    return "very_bad" if struggle > 0.55 else "bad"
                return "neutral"

            # Medium teach readiness: explanation/summary/fix can be good if affect/accuracy are reasonable.
            if 0.45 <= teach_ready < 0.75:
                if action == "ask_explanation":
                    return "very_good" if (mot > 0.55 and acc > 0.55) else "good"
                if action == "ask_summary":
                    return "good" if (emo > 0.45 and ret > 0.40) else "neutral"
                if action == "show_mistake_and_ask_fix":
                    return "good" if (acc > 0.55 and iq > 0.50) else "neutral"
                if action == "ask_worked_example":
                    # at this stage, asking for worked example gives less learner-side benefit
                    return "neutral"
                return "neutral"

            # High teach readiness: mistake-fixing is strongest protégé-style consolidation.
            # But if motivation/emotion are low, explanation is safer.
            if action == "show_mistake_and_ask_fix":
                return "very_good" if (mot > 0.45 and emo > 0.40 and acc > 0.55) else "good"
            if action == "ask_explanation":
                return "very_good" if mot > 0.50 else "good"
            if action == "ask_summary":
                return "good" if ret > 0.45 else "neutral"
            if action == "ask_worked_example":
                return "neutral" if struggle > 0.65 else "bad"
            return "neutral"

        return "neutral"

    def _apply_quality_shifts(self, topic_id: int, mode: str, q: str, dyn: TopicDynamics, prereq: float) -> None:
        """
        Category-based shifts of ALL learner variables (topic-specific dynamics).
        This is the key structural alignment with the paper's simulation design.
        """
        # base magnitude by quality level
        # positive gains reduce with difficulty; negative effects increase with difficulty
        if q == "very_good":
            base = 0.16 / dyn.difficulty
            emo_delta = 0.04
            mot_delta = 0.02
            acc_delta = 0.02
        elif q == "good":
            base = 0.09 / dyn.difficulty
            emo_delta = 0.02
            mot_delta = 0.01
            acc_delta = 0.01
        elif q == "neutral":
            base = 0.02 / dyn.difficulty
            emo_delta = 0.00
            mot_delta = 0.00
            acc_delta = 0.00
        elif q == "bad":
            base = -0.05 * dyn.difficulty
            emo_delta = -0.02 - dyn.impatience
            mot_delta = -0.02 - dyn.impatience
            acc_delta = -0.01
        else:  # very_bad
            base = -0.09 * dyn.difficulty
            emo_delta = -0.04 - 2.0 * dyn.impatience
            mot_delta = -0.04 - 2.0 * dyn.impatience
            acc_delta = -0.02

        # apply prereq scaling (readiness)
        base *= (0.40 + 0.60 * prereq)

        # stochastic noise
        n = dyn.noise_std
        noise = lambda: self.rng.gauss(0.0, n)

        # shorthand
        M = self.state.mastery_learner[topic_id]
        S = self.state.score[topic_id]
        ts = self.state.test_speed[topic_id]
        ss = self.state.segment_speed[topic_id]
        emo = self.state.emotion[topic_id]
        iq = self.state.input_quality[topic_id]

        # mastery and score improve with diminishing returns when near 1
        if base >= 0:
            dM = base * (1.0 - M) + noise()
            dS = (base * 0.9) * (1.0 - S) + noise()
            dts = (base * 0.6) * (1.0 - ts) + noise()
            dss = (base * 0.6) * (1.0 - ss) + noise()
            diq = (base * 0.7) * (1.0 - iq) + noise()
        else:
            # negative base: push down more when variables are already low (fragility)
            dM = base * (0.30 + 0.70 * M) + noise()
            dS = (base * 0.8) * (0.30 + 0.70 * S) + noise()
            dts = (base * 0.4) * (0.30 + 0.70 * ts) + noise()
            dss = (base * 0.4) * (0.30 + 0.70 * ss) + noise()
            diq = (base * 0.6) * (0.30 + 0.70 * iq) + noise()

        # protégé effect: when mode == tutee and quality is positive, learner gets extra mastery/retention boost
        protege_bonus = 0.0
        if mode == "tutee":
            if q in ("good", "very_good"):
                protege_bonus = 0.04 if q == "very_good" else 0.02

        # update topic variables
        self.state.mastery_learner[topic_id] = _clip01(M + dM + protege_bonus)
        self.state.score[topic_id] = _clip01(S + dS)
        self.state.test_speed[topic_id] = _clip01(ts + dts)
        self.state.segment_speed[topic_id] = _clip01(ss + dss)
        self.state.input_quality[topic_id] = _clip01(iq + diq)

        # emotion and motivation/global accuracy
        self.state.emotion[topic_id] = _clip01(emo + emo_delta + noise())
        self.state.motivation = _clip01(self.state.motivation + mot_delta + noise() * 0.5)
        self.state.accuracy = _clip01(self.state.accuracy + acc_delta + noise() * 0.3)

        # retention grows slowly with positive learning (and especially through teaching)
        if dM + protege_bonus > 0:
            self.state.retention = _clip01(self.state.retention + 0.03 * (dM + protege_bonus))
        else:
            # slight forgetting / instability under poor intervention
            self.state.retention = _clip01(self.state.retention + 0.01 * dM)

        # tutee mastery update: improves mainly when mode == tutee and learner performs well
        if mode == "tutee":
            MT = self.state.mastery_tutee[topic_id]
            if q == "very_good":
                dT = 0.10 * (1.0 - MT) + noise()
            elif q == "good":
                dT = 0.06 * (1.0 - MT) + noise()
            elif q == "neutral":
                dT = 0.02 * (1.0 - MT) + noise()
            else:
                dT = 0.01 * (1.0 - MT) + noise()  # still learns a bit
            self.state.mastery_tutee[topic_id] = _clip01(MT + dT)

    # --------------- reward & termination ----------------

    def _compute_reward(self, prev: dict, cur: LearnerTuteeState, topic_id: int) -> float:
        """
        Mastery-aligned reward.

        Key changes vs the previous version:
        - Reward is computed primarily from *selected topic* learning progress (mastery/score),
          rather than averaging percent changes across all topics and all variables.
        - Small shaping terms keep your multi-variable state meaningful (motivation/retention/accuracy),
          but they cannot dominate mastery gains.
        - Includes a small per-step penalty to discourage 500-step episodes without completion.
        """
        tid = int(topic_id)

        # --- primary learning signal (selected topic) ---
        d_mastery = float(cur.mastery_learner[tid] - prev["mastery_learner"][tid])
        d_score = float(cur.score[tid] - prev["score"][tid])

        # --- lightweight shaping (global) ---
        d_acc = float(cur.accuracy - prev["accuracy"])
        d_mot = float(cur.motivation - prev["motivation"])
        d_ret = float(cur.retention - prev["retention"])

        # optional: very small shaping from per-topic process variables
        # (kept small so they cannot "farm" reward without learning)
        d_emotion = _pct_change(float(cur.emotion[tid]), float(prev["emotion"][tid]))
        d_inputq = _pct_change(float(cur.input_quality[tid]), float(prev["input_quality"][tid]))

        # Weighted sum: mastery dominates
        reward = (
            2.0 * d_mastery
            + 0.6 * d_score
            + 0.2 * d_acc
            + 0.10 * d_mot
            + 0.10 * d_ret
            + 0.02 * d_emotion
            + 0.02 * d_inputq
        )

        # completion bonus (only when this topic flips to done)
        # if (not prev["topic_done"]) and bool(self.state.topic_done[tid]):
        #     reward += self.completion_bonus

        if all(cur.topic_done):
            reward += self.episode_success_bonus

        # discourage long episodes that never finish
        reward -= self.step_cost  # per environment step


        if cur.step_count >= self.max_steps:
            reward -= self.timeout_penalty

        # diminishing returns: too much assistance should reduce reward
        decay = 1.0 / (1.0 + self.diminishing_k * max(0, cur.assist_count))
        reward *= decay

        # clip reward
        lo, hi = self.reward_clip
        return max(lo, min(hi, reward))

    def _check_done(self) -> bool:
        # end when all topics are complete
        if all(self.state.topic_done):
            return True
        # safety cap
        if self.state.step_count >= self.max_steps:
            return True
        return False
