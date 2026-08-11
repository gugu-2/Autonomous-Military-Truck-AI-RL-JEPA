"""JEPA Latent Predictor — predicts future latent states from current context."""

import torch
import torch.nn as nn


class TemporalPositionEmbedding(nn.Module):
    def __init__(self, prediction_horizon: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(prediction_horizon, embed_dim)

    def forward(self, k: int) -> torch.Tensor:
        # k is the forward step offset
        return self.embed(torch.tensor(k, device=self.embed.weight.device))


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        depth=6,
        num_heads=8,
        prediction_horizon=10,
        mlp_ratio=4.0,
        action_dim: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.prediction_horizon = prediction_horizon

        self.temporal_embed = TemporalPositionEmbedding(prediction_horizon, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self, context_tokens: torch.Tensor, action_context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Input: context_tokens (B, N, D)
        Output: predicted futures (B, K, N, D) where K is prediction_horizon
        """
        B, N, D = context_tokens.shape
        predictions = []

        actions = action_context
        if actions is not None:
            if actions.dim() == 2:
                actions = actions.unsqueeze(1).expand(-1, self.prediction_horizon, -1)

        for k in range(self.prediction_horizon):
            step_action = actions[:, k, :] if actions is not None else None
            pred = self.predict_single_step(context_tokens, k, step_action)
            predictions.append(pred.unsqueeze(1))

        return torch.cat(predictions, dim=1)  # (B, K, N, D)

    def predict_single_step(
        self, s_t: torch.Tensor, k: int, action_context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict one step ahead."""
        t_emb = self.temporal_embed(k)
        x = s_t + t_emb

        if action_context is not None:
            a_emb = self.action_proj(action_context).unsqueeze(1)  # (B, 1, D)
            x = x + a_emb

        pred = self.transformer(x)
        return pred
