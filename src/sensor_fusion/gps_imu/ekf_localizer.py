"""15-DOF Extended Kalman Filter for GPS/IMU sensor fusion localization."""
import numpy as np
from src.sensor_fusion.fusion.unified_world_state import EgoPose

class EKFLocalizer:
    """15-DOF Extended Kalman Filter for localization."""
    
    def __init__(self, config: dict):
        # State vector: [px, py, pz, vx, vy, vz, roll, pitch, yaw, bax, bay, baz, bgx, bgy, bgz]
        self.x = np.zeros(15)
        
        # Covariance matrix P
        self.P = np.eye(15) * 1.0
        
        # Process noise Q
        self.Q = np.eye(15) * 0.1
        if 'Q_diag' in config:
            np.fill_diagonal(self.Q, config['Q_diag'])
            
        # Measurement noise R (GPS)
        self.R_gps = np.eye(3) * 2.0
        if 'R_gps_diag' in config:
            np.fill_diagonal(self.R_gps, config['R_gps_diag'])
            
        # Measurement noise R (Velocity)
        self.R_vel = np.eye(3) * 0.5
        
        # Gravity
        self.g = np.array([0, 0, -9.81])
        
    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> EgoPose:
        """IMU prediction step."""
        # Extract states
        p = self.x[0:3]
        v = self.x[3:6]
        rpy = self.x[6:9]
        ba = self.x[9:12]
        bg = self.x[12:15]
        
        # Correct IMU measurements with bias
        accel_corrected = accel - ba
        gyro_corrected = gyro - bg
        
        # Rotation matrix from Euler angles (roll, pitch, yaw)
        cr, cp, cy = np.cos(rpy)
        sr, sp, sy = np.sin(rpy)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        
        # Transform acceleration to world frame and add gravity
        accel_world = R @ accel_corrected + self.g
        
        # Update states (kinematic equations)
        p_new = p + v * dt + 0.5 * accel_world * dt**2
        v_new = v + accel_world * dt
        rpy_new = rpy + gyro_corrected * dt  # Approximation for small dt
        
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:9] = rpy_new
        
        # Jacobian F (simplified)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        # Detailed Jacobian components omitted for brevity, but would include derivatives of R w.r.t rpy
        
        # Update Covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return self.get_pose()
        
    def update_gps(self, gps_pos: np.ndarray, gps_cov: np.ndarray):
        """GPS measurement update step."""
        # Measurement matrix H (we measure px, py, pz directly)
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)
        
        # Measurement residual
        y = gps_pos - self.x[0:3]
        
        # Innovation covariance
        S = H @ self.P @ H.T + gps_cov
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(15) - K @ H) @ self.P
        
    def update_velocity(self, vel: np.ndarray):
        """Velocity update from wheel odometry or radar."""
        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)
        
        y = vel - self.x[3:6]
        S = H @ self.P @ H.T + self.R_vel
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(15) - K @ H) @ self.P
        
    def get_pose(self) -> EgoPose:
        """Returns current best estimate as EgoPose."""
        return EgoPose(
            x=float(self.x[0]), y=float(self.x[1]), z=float(self.x[2]),
            velocity_x=float(self.x[3]), velocity_y=float(self.x[4]), velocity_z=float(self.x[5]),
            roll=float(self.x[6]), pitch=float(self.x[7]), yaw=float(self.x[8]),
            acceleration_x=0.0, acceleration_y=0.0, acceleration_z=0.0 # Accel would be derived
        )
