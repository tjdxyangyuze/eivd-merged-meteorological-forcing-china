"""
EIVDTemperatureFusion: Temperature-specific EIVD fusion with unit conversion and topography.
"""

import numpy as np
from .eivd_base import EIVDFusionBase


class EIVDTemperatureFusion(EIVDFusionBase):
    """
    Temperature fusion with Celsius-to-Kelvin conversion and physical constraints.

    Unit: Kelvin (K)
    Physical range: 200-350 K [Eq.(9)]
    Correction limit: 20 K
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_range = (200.0, 350.0)
        self.correction_limit = 20.0
        self.unit = 'K'
        self.lapse_rate_limit = 0.01

    def preprocess_data(self, data, data_type=None):
        """
        Preprocess temperature data: unit conversion and range check.

        Parameters
        ----------
        data : ndarray
            Raw temperature data.
        data_type : str, optional
            Source type (not used for temperature but kept for interface).

        Returns
        -------
        ndarray
            Preprocessed data in Kelvin.
        """
        if data is None:
            return None

        # Convert Celsius to Kelvin if mean value is below 150
        if not np.all(np.isnan(data)):
            if np.nanmean(data) < 150:
                data = data + 273.15

        # Physical range check
        data[(data < self.valid_range[0]) | (data > self.valid_range[1])] = np.nan

        return data

    def apply_correction_limit(self, bias):
        """Apply variable-specific correction limit."""
        return np.clip(bias, -self.correction_limit, self.correction_limit)

    def apply_topographic_correction(self, merged_data, dem_data,
                                     station_elev, station_obs,
                                     station_lons, station_lats,
                                     lon_grid, lat_grid):
        """
        Apply topographic regression correction for temperature.

        Uses a standard atmospheric lapse rate constraint to limit over-correction.
        """
        if dem_data is None or station_elev is None or len(station_elev) < 3:
            return merged_data

        # Extract merged values at station locations
        merged_at_stations = []
        valid_station_indices = []

        for idx, (slon, slat) in enumerate(zip(station_lons, station_lats)):
            i = int((self.lat_max - slat) / self.resolution + 0.5)
            j = int((slon - self.lon_min) / self.resolution + 0.5)

            if 0 <= i < self.nrows and 0 <= j < self.ncols:
                if not np.isnan(merged_data[i, j]):
                    merged_at_stations.append(merged_data[i, j])
                    valid_station_indices.append(idx)

        if len(valid_station_indices) < 3:
            return merged_data

        station_elev_valid = station_elev[valid_station_indices]
        station_obs_valid = station_obs[valid_station_indices]
        merged_at_stations = np.array(merged_at_stations)

        # Residuals: observation - merged value
        residuals = station_obs_valid - merged_at_stations

        valid_mask = ~np.isnan(residuals) & ~np.isnan(station_elev_valid)
        if np.sum(valid_mask) < 3:
            return merged_data

        x_reg = station_elev_valid[valid_mask]
        y_reg = residuals[valid_mask]

        x_mean = np.mean(x_reg)
        y_mean = np.mean(y_reg)
        numerator = np.sum((x_reg - x_mean) * (y_reg - y_mean))
        denominator = np.sum((x_reg - x_mean) ** 2)

        if denominator > 0:
            slope = numerator / denominator
            # Constrain lapse rate to physically realistic range
            slope = np.clip(slope, -self.lapse_rate_limit, self.lapse_rate_limit)
            intercept = y_mean - slope * x_mean

            # Apply correction
            corrected = merged_data + slope * (dem_data - np.mean(x_reg))
            # Limit correction magnitude
            corrected = np.clip(corrected, merged_data - 5, merged_data + 5)
            return np.clip(corrected, self.valid_range[0], self.valid_range[1])

        return merged_data
