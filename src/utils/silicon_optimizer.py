import torch


class M2Optimizer:
    @staticmethod
    def get_optimal_batch_size(total_samples: int) -> int:
        """Determine optimal batch size for M2 processor."""
        if torch.backends.mps.is_available():
            available_mem = 8 * 1024 * 1024 * 1024  # Assuming 8GB available memory
            return min(32, total_samples)  # Conservative batch size for M2
        else:
            return min(64, total_samples)

    @staticmethod
    def optimize_memory_usage():
        """Apply memory optimizations for M2."""
        if torch.backends.mps.is_available():
            # Clear GPU memory cache
            torch.mps.empty_cache()

            # Set memory efficient attention if available
            torch._C._jit_set_profiling_mode(False)

    @staticmethod
    def get_device() -> torch.device:
        """Get the optimal device for the current system."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
