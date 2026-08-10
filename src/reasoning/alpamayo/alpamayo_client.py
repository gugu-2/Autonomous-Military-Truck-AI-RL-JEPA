import torch
import logging
from typing import Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

class AlpamayoClient:
    """
    Client for interacting with the Vision-Language-Action (VLA) model (Alpamayo/LLaVA).
    Runs asynchronously on a separate CUDA stream to not block the main 12ms control loop.
    """
    def __init__(self, config: Dict[str, Any]):
        self.device = config.get('device', 'cuda')
        self.model_id = config.get('model_id', 'llava-hf/llava-v1.6-mistral-7b-hf')
        
        self.processor = None
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Lazy load the massive 7B model using 4-bit quantization."""
        if self.is_loaded:
            return
            
        try:
            from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
            
            logger.info(f"Loading VLA model {self.model_id} in 4-bit precision...")
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.model.eval()
            self.is_loaded = True
            logger.info("VLA model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load VLA model: {e}")
            self.is_loaded = False

    @torch.no_grad()
    def generate_action_hint(self, image: Image.Image, prompt: str) -> str:
        """
        Runs a forward pass through the VLA model to reason about a scene.
        """
        if not self.is_loaded:
            self.load_model()
            if not self.is_loaded:
                return "ERROR: Model not loaded."
                
        formatted_prompt = f"[INST] <image>\n{prompt} [/INST]"
        
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt").to(self.device)
        
        # Run inference on a separate CUDA stream for async execution
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2, # Low temperature for deterministic actions
                do_sample=False
            )
            
        response = self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
