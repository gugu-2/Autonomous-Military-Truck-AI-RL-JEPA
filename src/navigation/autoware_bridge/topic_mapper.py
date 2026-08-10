class TopicMapper:
    """
    Maintains a mapping of OMNIDRIVE internal data types to standard Autoware ROS 2 topics.
    This allows easy replacement of Autoware modules with OMNIDRIVE AI modules.
    """

    # Standard Autoware topic structure
    TOPICS = {
        # Inputs to Autoware (What OMNIDRIVE provides)
        "perception_objects": "/perception/object_recognition/objects",
        "localization_pose": "/localization/kinematic_state",
        "ego_vehicle_status": "/vehicle/status/velocity_status",
        # Outputs from Autoware (What OMNIDRIVE consumes)
        "hd_map_data": "/map/vector_map",
        "route": "/planning/mission_planning/route",
        "traffic_signals": "/perception/traffic_light_recognition/traffic_signals",
        "trajectory": "/planning/trajectory",
    }

    @classmethod
    def get_ros_type(cls, topic_key: str):
        """
        Returns the appropriate ROS 2 message type class for a given topic.
        Requires rclpy and autoware_auto_msgs.
        """
        try:
            from autoware_auto_perception_msgs.msg import PredictedObjects, TrafficSignalArray
            from autoware_auto_planning_msgs.msg import Trajectory
            from nav_msgs.msg import Odometry

            types = {
                "perception_objects": PredictedObjects,
                "traffic_signals": TrafficSignalArray,
                "localization_pose": Odometry,
                "trajectory": Trajectory,
            }
            return types.get(topic_key)

        except ImportError:
            # Fallback/mock for environments without ROS 2 installed
            return None
