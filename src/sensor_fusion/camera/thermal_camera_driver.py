import numpy as np

class ThermalCameraDriver:
    """Hardware interface for long-wave infrared (LWIR) thermal cameras used in military setups."""
    def get_frame(self) -> np.ndarray:
        # Mock returning a thermal image (1-channel or pseudocolor 3-channel)
        return np.random.randint(0, 255, (224, 224, 1), dtype=np.uint8)
