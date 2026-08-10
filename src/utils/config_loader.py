import os
from pathlib import Path
from typing import Dict, Optional, Any

from omegaconf import DictConfig, OmegaConf
from .common_types import VehicleMode

class ConfigurationError(Exception):
    """Raised when configuration loading fails."""
    pass

_GLOBAL_CONFIG: Optional[DictConfig] = None

def load_config(vehicle_mode: VehicleMode, overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """
    Loads base_config.yaml, then merges vehicle-specific config.
    """
    global _GLOBAL_CONFIG
    try:
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config"
        
        base_config_path = config_dir / "base_config.yaml"
        if not base_config_path.exists():
            # Provide an empty one or default structure if not found for robustness
            base_config = OmegaConf.create({})
        else:
            base_config = OmegaConf.load(base_config_path)
            
        vehicle_config_path = config_dir / f"{vehicle_mode.value}_config.yaml"
        if vehicle_config_path.exists():
            vehicle_config = OmegaConf.load(vehicle_config_path)
            config = OmegaConf.merge(base_config, vehicle_config)
        else:
            config = base_config
            
        if overrides:
            override_config = OmegaConf.create(overrides)
            config = OmegaConf.merge(config, override_config)
            
        _GLOBAL_CONFIG = config
        return config
    except Exception as e:
        raise ConfigurationError(f"Failed to load config for {vehicle_mode}: {e}") from e

def get_config() -> DictConfig:
    """
    Returns the singleton config instance.
    """
    if _GLOBAL_CONFIG is None:
        raise ConfigurationError("Config has not been loaded. Call load_config first.")
    return _GLOBAL_CONFIG

def save_config(config: DictConfig, path: str) -> None:
    """
    Saves the config to a YAML file.
    """
    try:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            OmegaConf.save(config=config, f=f)
    except Exception as e:
        raise ConfigurationError(f"Failed to save config to {path}: {e}") from e

__all__ = [
    'ConfigurationError',
    'load_config',
    'get_config',
    'save_config',
]
