"""
Learner Model
-------------
A simplified synthetic learner used for the hierarchical RL tutoring system.

The learner maintains a small set of interpretable state variables:
- mastery:    how well the learner understands the topic (0–1)
- motivation: how willing the learner is to continue effort (0–1)
- error_rate: likelihood of making mistakes (0–1)
- retention:  long-term consolidation of knowledge (0–1)

The learner is updated after:
1. Tutor actions (e.g., hint, worked example, reflection prompt)
2. Tutee interactions (Protégé Effect → deeper processing)

This module does NOT simulate real humans — it simulates *qualitative*
learning dynamics consistent with educational psychology.
"""

import numpy as np


class Learner:
    def __init__(
        self,
        mastery: float = 0.2,
        motivation: float = 0.7,
        error_rate: float = 0.6,
        retention: float = 0.1,
        noise_std: float = 0.02,
    ):
        """
        Initialize the learner state.
        Noise is added to updates so that each learner seed is slightly different.
        """
        self.mastery = mastery
        self.motivation = motivation
        self.error_rate = error_rate
        self.retention = retention
        self.noise_std = noise_std

    # ----------------------------------------------------------------------
    # Utility functions
    # ----------------------------------------------------------------------

    def _clip(self):
        """Ensure all values remain in [0, 1]."""
        self.mastery = np.clip(self.mastery, 0.0, 1.0)
        self.motivation = np.clip(self.motivation, 0.0, 1.0)
        self.error_rate = np.clip(self.error_rate, 0.0, 1.0)
        self.retention = np.clip(self.retention, 0.0, 1.0)

    def _noise(self):
        """Small Gaussian noise added to updates (realistic variation)."""
        return np.random.normal(0, self.noise_std)

    # ----------------------------------------------------------------------
    # Tutor Interaction Updates
    # ----------------------------------------------------------------------

    def update_after_tutor_action(self, action: str):
        """
        Apply effects of tutor actions.
        The magnitudes are chosen to be small and realistic in simulation.
        """

        if action == "hint":
            self.mastery += 0.03 + self._noise()
            self.error_rate -= 0.03 + self._noise()

        elif action == "worked_example":
            self.mastery += 0.07 + self._noise()
            self.motivation += 0.03 + self._noise()
            self.error_rate -= 0.05 + self._noise()

        elif action == "reflection_question":
            self.mastery += 0.04 + self._noise()
            self.retention += 0.03 + self._noise()

        elif action == "no_help":
            # learner becomes discouraged when struggling
            self.motivation -= 0.03 + self._noise()
            self.error_rate += 0.03 + self._noise()

        else:
            # Unknown action – no effect
            pass

        self._clip()

    # ----------------------------------------------------------------------
    # Tutee Interaction Updates (Your Thesis Contribution)
    # ----------------------------------------------------------------------

    def update_after_tutee_interaction(self, response_quality: str):
        """
        response_quality ∈ {"correct", "partial", "incorrect"}

        This simulates the Protégé Effect:
        - explaining helps mastery & motivation
        - incorrect explanations reveal gaps
        """

        if response_quality == "correct":
            self.mastery += 0.05 + self._noise()
            self.motivation += 0.04 + self._noise()
            self.error_rate -= 0.04 + self._noise()
            self.retention += 0.05 + self._noise()

        elif response_quality == "partial":
            self.mastery += 0.02 + self._noise()
            self.motivation += 0.02 + self._noise()
            self.retention += 0.02 + self._noise()

        elif response_quality == "incorrect":
            self.motivation -= 0.03 + self._noise()
            self.error_rate += 0.05 + self._noise()

        else:
            # no effect if unknown keyword
            pass

        self._clip()

    # ----------------------------------------------------------------------
    # State Access
    # ----------------------------------------------------------------------

    def get_state(self):
        """
        Return a compact representation usable by RL agents.
        You may discretize it later (optional).
        """
        return (
            round(self.mastery, 3),
            round(self.motivation, 3),
            round(self.error_rate, 3),
            round(self.retention, 3),
        )

    def __repr__(self):
        return (
            f"Learner(mastery={self.mastery:.2f}, "
            f"motivation={self.motivation:.2f}, "
            f"error_rate={self.error_rate:.2f}, "
            f"retention={self.retention:.2f})"
        )
