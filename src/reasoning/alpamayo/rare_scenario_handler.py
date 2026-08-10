import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class RareScenarioHandler:
    """
    Routes the structured intent from the VLA model to the appropriate control subsystem.
    """

    def __init__(self):
        self.active_scenario = None
        self.override_active = False

    def handle_intent(
        self,
        intent: dict[str, Any],
        rl_policy_callback: Callable,
        emergency_brake_callback: Callable,
    ):
        """
        Takes the structured intent and injects it into the driving policy.
        """
        action = intent.get("action")
        reason = intent.get("reason")

        if action == "STOP":
            logger.warning(f"VLA Reasoner requested STOP: {reason}")
            # Trigger safe stop (not necessarily emergency AEB unless immediate)
            emergency_brake_callback(reason=f"VLA: {reason}")
            self.override_active = True

        elif action == "SLOW_DOWN":
            logger.info(f"VLA Reasoner requested SLOW_DOWN: {reason}")
            # Hint the RL policy to reduce speed
            rl_policy_callback(action_mask={"throttle": 0.3})  # Limit max throttle
            self.override_active = True

        elif action == "REROUTE":
            logger.info(f"VLA Reasoner requested REROUTE: {reason}")
            # This would interface with Layer 6 (Autoware Navigation)
            pass

        elif action == "PROCEED":
            # Normal driving, clear overrides
            if self.override_active:
                logger.info("VLA Reasoner cleared overrides. Normal driving resumed.")
                self.override_active = False
        else:
            logger.warning(f"Unknown VLA action: {action}")
