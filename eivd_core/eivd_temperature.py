import numpy as np
from .eivd_base import EIVDFusionBase


class EIVDTemperatureFusion(EIVDFusionBase):
    """
    EIVD fusion for near-surface temperature.

    Unit: Kelvin (K)
    Physical range: 200-350 K
    Correction limit: 20 K
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_range = (200.0, 350.0)
        self.correction_limit = 20.0
        self.unit = 'K'

    def preprocess_data(self, data):
        if data is None:
            return None

        # Convert Celsius to Kelvin if needed
        if not np.all(np.isnan(data)):
            if np.nanmean(data) < 150:
                data = data + 273.15

        # Physical range check
        data[(data < self.valid_range[0]) | (data > self.valid_range[1])] = np.nan

        return data

    def apply_correction_limit(self, bias):
        return np.clip(bias, -self.correction_limit, self.correction_limit)
