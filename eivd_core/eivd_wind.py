"""
EIVDWindFusion: Wind-specific EIVD fusion with two-step decomposition [Eq.(3)].
"""

import numpy as np
from .eivd_base import EIVDFusionBase


class EIVDWindFusion(EIVDFusionBase):
    """
    Wind fusion using two-step strategy.

    Step 1: Merge wind speed magnitude [Eq.(8)]
    Step 2: Decompose U/V components using station ratios [Eq.(3)]

    Validation uses only wind speed magnitude.
    Physical range: 0-50 m/s
    Correction limit: 5 m/s
    """

    def __init__(self, use_two_step=True, **kwargs):
        super().__init__(**kwargs)
        self.valid_range = (0.0, 50.0)
        self.correction_limit = 5.0
        self.unit = 'm/s'
        self.use_two_step = use_two_step
        self.lapse_rate_limit = 0.05

    def preprocess_data(self, data, data_type=None):
        """
        Preprocess wind speed data: physical range check only.

        Parameters
        ----------
        data : ndarray
            Raw wind speed data.
        data_type : str, optional
            Source type (not used but kept for interface).

        Returns
        -------
        ndarray
            Preprocessed data in m/s.
        """
        if data is None:
            return None

        # Physical range check
        data[(data < self.valid_range[0]) | (data > self.valid_range[1])] = np.nan

        return data

    def apply_correction_limit(self, bias):
        """Apply variable-specific correction limit."""
        return np.clip(bias, -self.correction_limit, self.correction_limit)

    def decompose_uv(self, ws_merged, u_sta, v_sta, ws_sta):
        """
        Decompose wind speed into U and V components [Eq.(3)].

        Eq.(3): U_merged = WS_merged * (U_sta / WS_sta)
                V_merged = WS_merged * (V_sta / WS_sta)

        Parameters
        ----------
        ws_merged : ndarray
            Merged wind speed magnitude.
        u_sta, v_sta : ndarray
            Station U and V components.
        ws_sta : ndarray
            Station wind speed magnitude.

        Returns
        -------
        tuple
            (u_merged, v_merged) or (None, None) if decomposition is disabled.
        """
        if not self.use_two_step:
            return None, None

        # Avoid division by zero
        mask = ws_sta > 0.1
        u_merged = np.full_like(ws_merged, np.nan)
        v_merged = np.full_like(ws_merged, np.nan)

        u_merged[mask] = ws_merged[mask] * (u_sta[mask] / ws_sta[mask])
        v_merged[mask] = ws_merged[mask] * (v_sta[mask] / ws_sta[mask])

        return u_merged, v_merged

    def apply_topographic_correction(self, merged_data, dem_data,
                                     station_elev, station_obs,
                                     station_lons, station_lats,
                                     lon_grid, lat_grid):
        """
        Apply topographic regression correction for wind speed.

        Wind speed is influenced by terrain, but the relationship is complex.
        This implementation applies a conservative correction.
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
            corrected = np.clip(corrected, merged_data - 2, merged_data + 2)
            return np.clip(corrected, self.valid_range[0], self.valid_range[1])

        return merged_data
