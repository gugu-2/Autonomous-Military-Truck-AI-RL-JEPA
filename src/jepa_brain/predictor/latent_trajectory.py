import torch


class LatentTrajectory:
    """Stores the predicted sequence of future latent states."""

    def __init__(self):
        self.states: list[torch.Tensor] = []

    def append(self, latent_state: torch.Tensor):
        self.states.append(
            latent_state.detach() if isinstance(latent_state, torch.Tensor) else latent_state
        )

    def get_sequence(self) -> torch.Tensor:
        if not self.states:
            return torch.empty(0)
        return torch.stack(self.states, dim=1)

    def clear(self) -> None:
        """Reset trajectory buffer between rollouts."""
        self.states.clear()
