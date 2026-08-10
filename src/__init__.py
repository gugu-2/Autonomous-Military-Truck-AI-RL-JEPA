"""
OMNIDRIVE PROJECT
A 7-layer autonomous driving AI.
"""

__version__ = "1.0.0"
__author__ = "OmniDrive AI Team"

from .omnidrive_brain import OmniDriveBrain, VehicleMode

__all__ = ["OmniDriveBrain", "VehicleMode", "__version__"]
