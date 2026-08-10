import logging
from typing import Any

logger = logging.getLogger(__name__)


class CommandTranslator:
    """
    Translates structured NLP intents into actual RL reward modifications or action masks.
    This provides the bridge between language reasoning and mathematical continuous control.
    """

    def __init__(self):
        pass

    def generate_rl_hints(self, intent: dict[str, Any]) -> dict[str, Any]:
        """
        Converts the intent (e.g. SLOW_DOWN, YIELD) into mathematical constraints
        that the RL driving policy can apply immediately.
        """
        action = intent.get("action", "PROCEED")
        hints = {"action_mask": {}, "reward_modifier": {}}

        if action == "STOP":
            # Override completely, don't just hint
            hints["action_mask"] = {"throttle": 0.0, "brake": 1.0}
        elif action == "SLOW_DOWN":
            # Cap the maximum throttle
            hints["action_mask"] = {"throttle_max": 0.3}
            # Penalize speed
            hints["reward_modifier"] = {"speed_penalty": 5.0}
        elif action == "YIELD":
            # Similar to slow down but wait for clearance
            hints["action_mask"] = {"throttle_max": 0.1}
        elif action == "TURN_LEFT":
            hints["reward_modifier"] = {"lane_bias": -1.0}  # Bias towards left lane
        elif action == "TURN_RIGHT":
            hints["reward_modifier"] = {"lane_bias": 1.0}  # Bias towards right lane

        return hints
