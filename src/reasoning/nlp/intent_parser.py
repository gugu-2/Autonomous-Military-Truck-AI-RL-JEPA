import re
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class IntentParser:
    """
    Parses natural language output from the VLA model into structured JSON intent.
    """
    VALID_ACTIONS = ["STOP", "SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT", "REROUTE", "YIELD", "PROCEED"]
    
    def parse(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts action and reason from the raw model output.
        Expected format: "ACTION. Reason text..."
        Example: "STOP. A uniformed soldier is displaying a hand stop signal."
        """
        if not raw_text:
            return None
            
        # Try to find the action word at the beginning of the text
        text_upper = raw_text.upper()
        
        detected_action = "PROCEED" # Default safe fallback
        reason = raw_text
        
        for action in self.VALID_ACTIONS:
            # Look for exact action word as the first word
            if text_upper.startswith(action):
                detected_action = action
                # Extract the rest of the text as the reason
                reason_match = re.search(f"{action}[^a-zA-Z]*(.*)", raw_text, re.IGNORECASE)
                if reason_match:
                    reason = reason_match.group(1).strip()
                break
                
        # If no explicit action at the start, try to extract intent from keywords
        if detected_action == "PROCEED":
            if "stop" in text_upper:
                detected_action = "STOP"
            elif "slow" in text_upper:
                detected_action = "SLOW_DOWN"
            elif "reroute" in text_upper or "alternative" in text_upper:
                detected_action = "REROUTE"
                
        return {
            "action": detected_action,
            "reason": reason
        }
