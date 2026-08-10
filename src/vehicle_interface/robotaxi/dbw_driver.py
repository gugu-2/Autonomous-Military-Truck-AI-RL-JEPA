import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DBWDriver:
    """
    Drive-by-Wire (DBW) interface for Robotaxis.
    Translates normalized AI commands into vehicle-specific DBW commands 
    (e.g., Dataspeed, PACMod, or direct OEM CAN).
    """
    def __init__(self, config: Dict[str, Any]):
        self.dbw_type = config.get('dbw_type', 'dataspeed')
        self.max_steer_angle_rad = config.get('max_steer_angle_rad', 8.2) # typical steering wheel max
        
        self.enabled = False
        
    def enable(self):
        """Enable DBW control."""
        self.enabled = True
        logger.info(f"Drive-By-Wire ({self.dbw_type}) ENABLED.")
        
    def disable(self):
        """Disable DBW control, return to human driver."""
        self.enabled = False
        logger.info(f"Drive-By-Wire ({self.dbw_type}) DISABLED.")
        
    def send_commands(self, normalized_actions: Dict[str, float]) -> Dict[str, Any]:
        """
        Converts normalized [-1, 1] AI actions into raw DBW messages.
        """
        if not self.enabled:
            return {}
            
        steer_norm = normalized_actions.get('steering', 0.0)
        throttle_norm = normalized_actions.get('throttle', 0.0)
        brake_norm = normalized_actions.get('brake', 0.0)
        
        dbw_msgs = {}
        
        if self.dbw_type == 'dataspeed':
            # Dataspeed DBW format (simplified)
            dbw_msgs['steering_cmd'] = {
                'steering_wheel_angle_cmd': steer_norm * self.max_steer_angle_rad,
                'steering_wheel_angle_velocity': 0.0, # 0 = max speed
                'enable': True,
                'clear': False,
                'ignore': False
            }
            dbw_msgs['throttle_cmd'] = {
                'pedal_cmd': throttle_norm,
                'pedal_cmd_type': 2, # 2 = Percent (0.0 to 1.0)
                'enable': True
            }
            dbw_msgs['brake_cmd'] = {
                'pedal_cmd': brake_norm,
                'pedal_cmd_type': 2, # 2 = Percent
                'enable': True
            }
            
        return dbw_msgs
