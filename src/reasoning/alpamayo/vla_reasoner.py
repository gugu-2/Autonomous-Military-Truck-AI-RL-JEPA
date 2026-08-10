import logging
from typing import Any

from PIL import Image

from reasoning.alpamayo.alpamayo_client import AlpamayoClient
from reasoning.alpamayo.scene_descriptor import SceneDescriptor
from reasoning.nlp.intent_parser import IntentParser

logger = logging.getLogger(__name__)


class VLAReasoner:
    """
    Main coordinator for the Reasoning Module.
    Combines the VLA Client, Scene Descriptor, and Intent Parser.
    """

    def __init__(self, config: dict[str, Any]):
        self.client = AlpamayoClient(config)
        self.descriptor = SceneDescriptor(config)
        self.parser = IntentParser()

    def process_scene(
        self, image: Image.Image, telemetry: dict[str, float], hazard_energy: float
    ) -> dict[str, Any] | None:
        """
        Process a scene through the VLA model and return a structured driving intent.
        """
        # 1. Generate text prompt
        prompt = self.descriptor.generate_prompt(telemetry, hazard_energy)

        # 2. Get raw text response from VLA model
        logger.info("Querying VLA model for scene reasoning...")
        raw_response = self.client.generate_action_hint(image, prompt)

        # 3. Parse text into structured intent
        intent = self.parser.parse(raw_response)

        if intent:
            logger.info(
                f"VLA Reasoning output: Action={intent['action']}, Reason={intent['reason']}"
            )

        return intent
