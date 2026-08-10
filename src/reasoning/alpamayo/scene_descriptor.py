from typing import Dict, Any

class SceneDescriptor:
    """
    Constructs contextual text prompts for the VLA model combining visual cues
    with telemetry and JEPA hazard map data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.vehicle_mode = config.get('vehicle_mode', 'robotaxi')
        
    def generate_prompt(self, telemetry: Dict[str, float], hazard_energy: float) -> str:
        """
        Creates the prompt that accompanies the camera image into the Alpamayo model.
        """
        speed_kph = telemetry.get('speed', 0.0) * 3.6
        
        base_prompt = f"The vehicle is traveling at {speed_kph:.1f} km/h. "
        
        if hazard_energy > 0.45:
            base_prompt += f"The world model detects a high anomaly/hazard score ({hazard_energy:.2f}). "
            
        base_prompt += "Based on the image, what is the rare or hazardous event occurring, and what action should the autonomous vehicle take?"
        
        # Add mode-specific context
        if self.vehicle_mode == 'military':
            base_prompt += " Consider military contexts such as checkpoints, off-road terrain, IED visual cues, or soldier hand signals."
        elif self.vehicle_mode == 'truck':
            base_prompt += " Consider heavy truck contexts such as weigh stations, low bridges, or loading docks."
            
        return base_prompt
