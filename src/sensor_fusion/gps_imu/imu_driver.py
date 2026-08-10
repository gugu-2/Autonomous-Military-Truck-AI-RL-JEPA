class IMUDriver:
    """Hardware interface for 9-DOF IMU."""

    def get_reading(self):
        # Mock acceleration and gyro
        return {"accel": [0.0, 0.0, 9.81], "gyro": [0.0, 0.0, 0.0]}
