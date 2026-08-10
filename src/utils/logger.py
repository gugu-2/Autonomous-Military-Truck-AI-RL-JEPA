import sys
import time
from functools import wraps
from pathlib import Path
from typing import Optional, Any

from loguru import logger
import torch

def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    vehicle_mode: Optional[str] = None
) -> "loguru.Logger":
    """
    Sets up a structured logger using loguru.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    if log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        log_file = path / f"{name}_{vehicle_mode if vehicle_mode else 'general'}.log"
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="50 MB",
            retention="10 days",
            compression="zip",
            serialize=True, # JSON structured logging
        )
    return logger


def log_latency(name: str):
    """
    Performance timer context manager/decorator that logs ms latency.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            latency = (end - start) * 1000
            logger.debug(f"{name} took {latency:.2f} ms")
            return result
        return wrapper
    return decorator


def log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """
    Logs shape, min, max, and mean of a tensor for debugging.
    """
    if tensor is None:
        logger.debug(f"Tensor '{name}' is None")
        return
    
    try:
        shape = tuple(tensor.shape)
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        t_mean = tensor.to(torch.float32).mean().item()
        
        logger.debug(
            f"Tensor '{name}': shape={shape}, min={t_min:.4f}, max={t_max:.4f}, mean={t_mean:.4f}"
        )
    except Exception as e:
        logger.error(f"Failed to log tensor stats for '{name}': {e}")


# Default module-level logger
omni_logger = logger

__all__ = [
    'setup_logger',
    'log_latency',
    'log_tensor_stats',
    'omni_logger',
]
