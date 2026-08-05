import numpy as np
from .eivd_base import EIVDFusionBase


class EIVDWindFusion(EIVDFusionBase):
    """
    EIVD fusion for wind speed (two-step method).

    Step 1: Merge wind speed magnitude [Eq.(8)]
    Step 2: Decompose U/V using station ratios [Eq.(3)]

    Validation uses only wind speed magnitude.
    """

    def __init__(self, use_two_step=True, **kwargs):
        super().__init__(**kwargs)
        self.valid_range = (0.0, 50.0)
        self.correction_limit = 5.0
        self.unit = 'm/s'
        self.use_two_step = use_two_step

    def preprocess_data(self, data):
        if data is None:
            return None

        # Physical range check
        data[(data < self.valid_range[0]) | (data > self.valid_range[1])] = np.nan

        return data

    def apply_correction_limit(self, bias):
        return np.clip(bias, -self.correction_limit, self.correction_limit)

    def decompose_uv(self, ws_merged, u_sta, v_sta, ws_sta):
        """
        Decompose wind speed into U and V components [Eq.(3)].
        """
        if not self.use_two_step:
            return None, None

        mask = ws_sta > 0.1
        u_merged = np.full_like(ws_merged, np.nan)
        v_merged = np.full_like(ws_merged, np.nan)

        u_merged[mask] = ws_merged[mask] * (u_sta[mask] / ws_sta[mask])
        v_merged[mask] = ws_merged[mask] * (v_sta[mask] / ws_sta[mask])

        return u_merged, v_merged
