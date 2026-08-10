from typing import Any

import torch


class TrajectorySampler:
    """
    Samples multiple imagined trajectories using the world model to find the optimal path.
    Used for planning in complex scenarios (e.g., Model Predictive Control over RL).
    """

    def __init__(self, agent, config: dict[str, Any]):
        self.agent = agent
        self.num_samples = config.get("num_trajectory_samples", 100)
        self.horizon = config.get("imagination_horizon", 15)

    @torch.no_grad()
    def sample_optimal_trajectory(
        self, start_state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample multiple action sequences, evaluate them in imagination, and pick the best one.
        Returns the optimal action sequence and its predicted values.
        """
        # Broadcast start state to num_samples
        batch_state = {
            k: v.repeat(self.num_samples, *[1] * (v.ndim - 1)) for k, v in start_state.items()
        }

        # Imagine trajectories
        features, actions, values = self.agent.imagine_trajectory(batch_state, self.horizon)

        # Sum values over the horizon for each trajectory to get total return
        total_returns = values.sum(dim=1)

        # Find the trajectory with the maximum return
        best_idx = torch.argmax(total_returns)

        best_actions = actions[best_idx]
        best_values = values[best_idx]

        return best_actions, best_values
