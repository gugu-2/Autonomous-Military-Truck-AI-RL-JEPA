class ScenarioClassifier:
    """
    Lightweight heuristic classifier that decides IF the heavy VLA model needs to be invoked.
    Runs at 20Hz synchronously.
    """
    def __init__(self, hazard_threshold: float = 0.45, policy_confidence_threshold: float = 0.6):
        self.hazard_threshold = hazard_threshold
        self.policy_confidence_threshold = policy_confidence_threshold
        
    def should_trigger_vla(self, hazard_energy: float, rl_confidence: float) -> bool:
        """
        Trigger VLA reasoning if:
        1. JEPA detects an anomaly (high hazard energy) AND
        2. RL policy is unsure what to do (low confidence)
        """
        # If the RL policy knows exactly what to do, don't waste time on VLA.
        # If the hazard energy is low, it's a normal driving situation.
        if hazard_energy > self.hazard_threshold and rl_confidence < self.policy_confidence_threshold:
            return True
            
        # Hard trigger for extremely high hazard, just in case
        if hazard_energy > 0.80:
            return True
            
        return False
