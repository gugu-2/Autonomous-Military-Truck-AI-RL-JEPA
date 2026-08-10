"""JEPA World Model — the core AI brain of OMNIDRIVE."""
import torch
import torch.nn as nn
from typing import List

from ..encoder.patch_embedder import CameraTokenizer
from ..encoder.vit_encoder import ViTEncoder
from ..predictor.jepa_predictor import JEPAPredictor
from ..predictor.hazard_energy import HazardEnergyComputer
from ..predictor.ema_target_encoder import EMATargetEncoder
from .imagination_engine import ImaginationEngine, DrivingAction, LatentState

class UnifiedWorldState:
    def __init__(self, images: torch.Tensor):
        self.images = images

class JEPAWorldModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        self.tokenizer = CameraTokenizer(
            img_size=config.get("img_size", 224),
            patch_size=config.get("patch_size", 16),
            in_channels=config.get("in_channels", 3),
            embed_dim=config.get("embed_dim", 512),
            num_cameras=config.get("num_cameras", 4)
        )
        
        self.online_encoder = ViTEncoder(
            embed_dim=config.get("embed_dim", 512),
            depth=config.get("enc_depth", 12),
            num_heads=config.get("enc_heads", 8)
        )
        
        self.target_encoder = EMATargetEncoder(
            online_encoder=self.online_encoder,
            momentum=config.get("ema_momentum", 0.996)
        )
        
        self.predictor = JEPAPredictor(
            embed_dim=config.get("embed_dim", 512),
            depth=config.get("pred_depth", 6),
            num_heads=config.get("pred_heads", 8),
            prediction_horizon=config.get("pred_horizon", 10)
        )
        
        self.hazard_computer = HazardEnergyComputer(
            veto_threshold=config.get("veto_threshold", 0.8),
            emergency_threshold=config.get("emergency_threshold", 1.5)
        )
        
        self.imagination_engine = ImaginationEngine(
            jepa_predictor=self.predictor,
            hazard_computer=self.hazard_computer,
            config=config
        )
        
        self.register_buffer("current_hazard_map", torch.zeros(1, 16, 16))
        
    def encode(self, world_state: UnifiedWorldState) -> LatentState:
        tokens = self.tokenizer(world_state.images)
        context = self.online_encoder(tokens)
        return LatentState(tokens=context)
        
    def imagine_and_filter(self, latent: LatentState, actions: List[DrivingAction]) -> List[DrivingAction]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = self.imagination_engine.imagine(latent, actions)
            return self.imagination_engine.get_safe_trajectories(result)
            
    def get_hazard_map(self) -> torch.Tensor:
        return self.current_hazard_map
        
    def training_step(self, batch: dict) -> dict:
        """JEPA loss computation for pretraining."""
        img_t = batch['img_t']
        img_fut = batch['img_fut'] 
        action = batch['action']
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            latent_t = self.encode(UnifiedWorldState(img_t))
            s_hat = self.predictor(latent_t.tokens, action)
            
            B, K, C, H, W = img_fut.shape
            img_fut_flat = img_fut.view(B*K, C, H, W)
            target_tokens = self.tokenizer(img_fut_flat.unsqueeze(1).repeat(1,4,1,1,1))
            s_target = self.target_encoder(target_tokens)
            s_target = s_target.view(B, K, -1, self.config.get("embed_dim", 512))
            
            energy = self.hazard_computer.compute(s_target, s_hat)
            loss = energy.mean()
            
        self.target_encoder.update()
        
        return {"loss": loss}
        
    def load_pretrained(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(ckpt['state_dict'], strict=False)
