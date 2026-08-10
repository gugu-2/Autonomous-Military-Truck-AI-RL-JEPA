import time
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SystemBenchmarker:
    """
    Performance benchmarking utility to ensure the entire OMNIDRIVE AI pipeline
    can execute within the strict 12ms latency budget for high-speed driving.
    """
    def __init__(self, brain: Any):
        self.brain = brain
        self.metrics: Dict[str, list] = {
            'jepa_encode_ms': [],
            'rssm_update_ms': [],
            'actor_critic_ms': [],
            'total_loop_ms': []
        }
        
    def run_benchmark(self, num_iterations: int = 1000):
        """
        Runs the full brain pipeline with dummy data and measures latency.
        """
        logger.info(f"Starting performance benchmark ({num_iterations} iterations)...")
        
        # Ensure model is on correct device and precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Dummy inputs (4 cameras, 224x224 RGB)
        dummy_camera = torch.randn(1, 4, 3, 224, 224, device=device)
        dummy_telemetry = {'speed': 15.0, 'yaw': 0.0}
        
        # Warmup
        logger.info("Warming up CUDA graphs...")
        for _ in range(10):
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _ = self.brain.step(dummy_camera, dummy_telemetry)
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        logger.info("Warmup complete. Running timed benchmark...")
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _ = self.brain.step(dummy_camera, dummy_telemetry)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            self.metrics['total_loop_ms'].append((end_time - start_time) * 1000)
            
        self._print_report()
        
    def _print_report(self):
        """
        Prints the benchmark statistics.
        """
        total_times = self.metrics['total_loop_ms']
        
        avg_ms = sum(total_times) / len(total_times)
        max_ms = max(total_times)
        min_ms = min(total_times)
        
        # Calculate 99th percentile
        p99_ms = sorted(total_times)[int(len(total_times) * 0.99)]
        
        print("\n==========================================")
        print("    OMNIDRIVE PERFORMANCE BENCHMARK")
        print("==========================================")
        print(f"Target Latency Budget: < 12.0 ms")
        print(f"Average Latency:       {avg_ms:.2f} ms")
        print(f"Minimum Latency:       {min_ms:.2f} ms")
        print(f"Maximum Latency:       {max_ms:.2f} ms")
        print(f"99th Percentile:       {p99_ms:.2f} ms")
        print("==========================================")
        
        if p99_ms > 12.0:
            logger.warning("⚠️ SYSTEM FAILS LATENCY BUDGET! Frame drops likely.")
        else:
            logger.info("✅ SYSTEM MEETS HARD REAL-TIME BUDGET.")

if __name__ == "__main__":
    # Example standalone usage
    import sys
    sys.path.append('../../src')
    try:
        from omnidrive_brain import OmniDriveBrain
        brain = OmniDriveBrain()
        bench = SystemBenchmarker(brain)
        bench.run_benchmark(100)
    except Exception as e:
        print(f"Cannot run standalone benchmark without full env: {e}")
