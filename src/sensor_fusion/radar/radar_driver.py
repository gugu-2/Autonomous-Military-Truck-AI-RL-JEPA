class RadarDriver:
    """Hardware interface for RADAR sensors."""
    def get_targets(self):
        # Mock returning radar tracks (distance, velocity)
        return [{"id": 1, "dist": 45.0, "vel": -10.0}]
