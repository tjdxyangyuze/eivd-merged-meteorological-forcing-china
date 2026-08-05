"""
EIVDHumidityFusion: Humidity-specific EIVD fusion with Tetens formula and topography.
"""

import numpy as np
from .eivd_base import EIVDFusionBase


class EIVDHumidityFusion(EIVDFusionBase):
    """
    Humidity fusion with MERRA-2 dew point conversion [Eq.(1)].

    Eq.(1): RH = 100 * es(Td) / es(T),  es(T) = 6.112 * exp(17.67*T / (T+243.5))
    Physical range: 0-100%
    Correction limit: 30%
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_range = (0.0, 100.0)
        self.correction_limit = 30.0
        self.unit = '%'
        self.lapse_rate_limit = 0.5

    def calculate_rh_from_td(self, temp, td):
        """
        Calculate relative humidity from temperature and dew point [Eq.(1)].

        Parameters
        ----------
        temp : ndarray
            Temperature (K or °C).
        td : ndarray
            Dew point temperature (K or °C).

        Returns
        -------
        ndarray
            Relative humidity (0-100%).
        """
        # Convert Kelvin to Celsius if needed
        if not np.all(np.isnan(temp)) and np.nanmean(temp) > 150:
            temp_c = temp - 273.15
        else:
            temp_c = temp

        if not np.all(np.isnan(td)) and np.nanmean(td) > 150:
            td_c = td - 273.15
        else:
            td_c = td

        # Saturation vapor pressure using Tetens formula
        es_temp = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
        es_td = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))

        rh = 100.0 * es_td / es_temp

        # Physical constraints
        rh = np.clip(rh, 0.0, 100.0)
        invalid = (temp_c < -100) | (td_c < -100) | (temp_c > 60) | (td_c > 60)
        rh[invalid] = np.nan

        return rh

    def preprocess_data(self, data, data_type=None, temp_data=None):
        """
        Preprocess humidity data.

        For MERRA-2, calculate RH from dew point temperature [Eq.(1)].
        For other sources, just apply physical range check.

        Parameters
        ----------
        data : ndarray
            Raw humidity or dew point data.
        data_type : str, optional
            Source type ('merra2' or others).
        temp_data : ndarray, optional
            Temperature data for MERRA-2 RH calculation.

        Returns
        -------
        ndarray
            Preprocessed data in percentage.
        """
        if data is None:
            return None

        # MERRA-2: calculate RH from dew point [Eq.(1)]
        if data_type == 'merra2' and temp_data is not None:
            data = self.calculate_rh_from_td(temp_data, data)

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
        Apply topographic regression correction for humidity.

        Humidity typically decreases with elevation, but the relationship is
        weaker than for temperature. This implementation applies a conservative
        correction with a limited lapse rate.
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
            corrected = np.clip(corrected, merged_data - 10, merged_data + 10)
            return np.clip(corrected, self.valid_range[0], self.valid_range[1])

        return merged_data
