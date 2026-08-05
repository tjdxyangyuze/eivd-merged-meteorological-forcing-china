import numpy as np
from .eivd_base import EIVDFusionBase


class EIVDHumidityFusion(EIVDFusionBase):
    """
    EIVD fusion for relative humidity.

    MERRA-2 RH is calculated from dew point using the Tetens formula [Eq.(1)].
    Physical range: 0-100%
    Correction limit: 30%
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_range = (0.0, 100.0)
        self.correction_limit = 30.0
        self.unit = '%'

    def calculate_rh_from_td(self, temp, td):
        """
        Calculate RH from temperature and dew point [Eq.(1)].

        RH = 100 * es(Td) / es(T)
        es(T) = 6.112 * exp(17.67 * T / (T + 243.5))
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

        # Saturation vapor pressure
        es_temp = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
        es_td = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))

        rh = 100.0 * es_td / es_temp

        # Physical constraints
        rh = np.clip(rh, 0.0, 100.0)
        rh[(temp_c < -100) | (td_c < -100) | (temp_c > 60) | (td_c > 60)] = np.nan

        return rh

    def preprocess_data(self, data, data_type=None, temp_data=None):
        if data is None:
            return None

        # MERRA-2: calculate RH from dew point [Eq.(1)]
        if data_type == 'merra2' and temp_data is not None:
            data = self.calculate_rh_from_td(temp_data, data)

        # Physical range check
        data[(data < self.valid_range[0]) | (data > self.valid_range[1])] = np.nan

        return data

    def apply_correction_limit(self, bias):
        return np.clip(bias, -self.correction_limit, self.correction_limit)
