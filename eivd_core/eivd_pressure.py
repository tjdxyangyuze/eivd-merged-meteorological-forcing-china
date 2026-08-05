import numpy as np
from .eivd_base import EIVDFusionBase


class EIVDPressureFusion(EIVDFusionBase):
    """
    EIVD fusion for surface pressure.

    Unit: hPa (converted from Pa for reanalysis products)
    Physical range: 400-1300 hPa
    Correction limit: 50 hPa
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_range = (400.0, 1300.0)
        self.correction_limit = 50.0
        self.unit = 'hPa'

        self.data_units = {
            'era5': 'Pa',
            'merra2': 'Pa',
            'jra3q': 'Pa',
            'calibration': 'hPa'
        }

    def preprocess_data(self, data, data_type=None):
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
        return np.clip(bias, -self.correction_limit, self.correction_limit)
