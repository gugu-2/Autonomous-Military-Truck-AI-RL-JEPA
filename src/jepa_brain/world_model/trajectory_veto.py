import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TrajectoryVetoSystem:
    """Evaluates the hazard energy of an imagined trajectory and vetoes if dangerous."""
    def __init__(self, veto_threshold=0.70, caution_threshold=0.45):
        self.veto_threshold = veto_threshold
        self.caution_threshold = caution_threshold
        
    def evaluate(self, hazard_energy: float) -> str:
        """Returns action: 'VETO', 'CAUTION', or 'SAFE'"""
        if hazard_energy >= self.veto_threshold:
            logger.critical(f"Trajectory VETOED! Hazard Energy: {hazard_energy:.2f}")
            return 'VETO'
        elif hazard_energy >= self.caution_threshold:
            logger.warning(f"Trajectory flagged for CAUTION. Hazard Energy: {hazard_energy:.2f}")
            return 'CAUTION'
        return 'SAFE'
