class GPSDriver:
    """Hardware interface for RTK-GPS."""

    def get_position(self):
        # Mock lat/lon/alt
        return {"lat": 37.7749, "lon": -122.4194, "alt": 10.0}
