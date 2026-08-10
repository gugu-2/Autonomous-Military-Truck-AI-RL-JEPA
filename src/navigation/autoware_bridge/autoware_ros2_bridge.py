import logging
from typing import Dict, Any, Callable
import threading

logger = logging.getLogger(__name__)

class AutowareROS2Bridge:
    """
    Bridge connecting the core Python AI stack to the Autoware ROS 2 ecosystem.
    Handles pub/sub communication for HD maps, traffic rules, and localization data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.node_name = config.get('ros_node_name', 'omnidrive_ai_bridge')
        self.namespace = config.get('ros_namespace', '/omnidrive')
        self.is_connected = False
        
        self.subscribers = {}
        self.publishers = {}
        
        self.lock = threading.Lock()
        
    def connect(self):
        """
        Initialize rclpy (ROS 2 Python) and spin up the node.
        """
        try:
            import rclpy
            from rclpy.node import Node
            
            if not rclpy.ok():
                rclpy.init()
                
            self.node = rclpy.create_node(self.node_name, namespace=self.namespace)
            self.is_connected = True
            
            # Spin the node in a separate thread so it doesn't block
            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()
            
            logger.info("✅ Successfully connected to Autoware ROS 2 framework.")
        except ImportError:
            logger.error("Failed to import rclpy. Are you in a ROS 2 environment?")
            self.is_connected = False
            
    def _spin(self):
        import rclpy
        try:
            rclpy.spin(self.node)
        except Exception as e:
            logger.error(f"ROS 2 Spin exception: {e}")
            
    def subscribe(self, topic: str, msg_type: Any, callback: Callable):
        """Subscribe to a ROS 2 topic."""
        if not self.is_connected:
            return
            
        with self.lock:
            sub = self.node.create_subscription(msg_type, topic, callback, 10)
            self.subscribers[topic] = sub
            
    def publish(self, topic: str, msg_type: Any, msg_data: Any):
        """Publish to a ROS 2 topic."""
        if not self.is_connected:
            return
            
        with self.lock:
            if topic not in self.publishers:
                self.publishers[topic] = self.node.create_publisher(msg_type, topic, 10)
            
            self.publishers[topic].publish(msg_data)
            
    def shutdown(self):
        """Clean shutdown of ROS 2 connection."""
        if self.is_connected:
            import rclpy
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            self.is_connected = False
            logger.info("ROS 2 bridge shutdown.")
