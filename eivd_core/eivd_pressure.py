"""
EIVDPressureFusion: Pressure-specific EIVD fusion with unit conversion and topography.
"""

import numpy as np
from .eivd_base import EIVDFusionBase


class EIVDPressureFusion(EIVDFusionBase):
    """
    Pressure fusion with Pa-to-hPa conversion and physical constraints.

    Unit: hPa (converted from Pa for reanalysis products)
    Physical range: 400-1300 hPa
    Correction limit: 50 hPa
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_range = (400.0, 1300.0)
        self.correction_limit = 50.0
        self.unit = 'hPa'
        self.lapse_rate_limit = 0.1

        # Unit information for conversion
        self.data_units = {
            'era5': 'Pa',
            'merra2': 'Pa',
            'jra3q': 'Pa',
            'calibration': 'hPa'
        }

    def preprocess_data(self, data, data_type=None):
        """
        Preprocess pressure data: unit conversion and range check.

        Parameters
        ----------
        data : ndarray
            Raw pressure data.
        data_type : str, optional
            Source type ('era5', 'merra2', 'jra3q', 'calibration').

        Returns
        -------
        ndarray
            Preprocessed data in hPa.
        """
        if data is None:
            return None

        # Convert Pa to hPa (reanalysis products only)
        if data_type is not None and data_type in self.data_units:
            if self.data_units[data_type] == 'Pa':
                data = data / 100.0

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
        Apply topographic regression correction for surface pressure.

        Surface pressure decreases with elevation following the hypsometric equation.
        This implementation applies a physically-constrained correction.
        """
        if dem_data is None or station_elev is None or len(station_elev) < 3:
            return merged_data

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
            slope = np.clip(slope, -self.lapse_rate_limit, self.lapse_rate_limit)

            corrected = merged_data + slope * (dem_data - np.mean(x_reg))
            corrected = np.clip(corrected, merged_data - 20, merged_data + 20)
            return np.clip(corrected, self.valid_range[0], self.valid_range[1])

        return merged_data
