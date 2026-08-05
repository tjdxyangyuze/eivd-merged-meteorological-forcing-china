"""
MonthlyBiasCorrector: Compute and apply additive monthly bias correction [Eq.(2)].
"""

import numpy as np
from tqdm import tqdm


class MonthlyBiasCorrector:
    """
    Bias correction using station-interpolated monthly means as reference.

    Eq.(2): X_corr = X + (M_sta - M_src)
    where M_sta is the calibration monthly mean and M_src is the source mean.
    """

    def __init__(self, start_date, end_date, calibration_start, calibration_end,
                 nrows, ncols, correction_limit=50.0, nodata_values=None):
        """
        Initialize the bias corrector.

        Parameters
        ----------
        start_date : datetime
            Start date of the study period.
        end_date : datetime
            End date of the study period.
        calibration_start : datetime
            Start date of the calibration period.
        calibration_end : datetime
            End date of the calibration period.
        nrows, ncols : int
            Grid dimensions.
        correction_limit : float
            Maximum allowed absolute bias correction value.
        nodata_values : list, optional
            Values to treat as no-data.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.nrows = nrows
        self.ncols = ncols
        self.correction_limit = correction_limit

        if nodata_values is None:
            nodata_values = [-9999, -9999.0, 65535, 32767, -32768]
        self.nodata_values = nodata_values

        self.bias_factors = None

    def compute_monthly_means(self, date_to_data_func, date_list):
        """
        Compute monthly means from daily data for each grid cell.

        Parameters
        ----------
        date_to_data_func : callable
            Function that accepts a date and returns a 2D numpy array.
        date_list : list of datetime
            List of dates to process.

        Returns
        -------
        dict
            Nested dictionary: {year: {month: 2D array of monthly means}}
        """
        sums = {}
        counts = {}

        for dt in tqdm(date_list, desc="Computing monthly means"):
            data = date_to_data_func(dt)
            if data is None:
                continue

            y, m = dt.year, dt.month

            if y not in sums:
                sums[y] = {}
                counts[y] = {}
            if m not in sums[y]:
                sums[y][m] = np.zeros((self.nrows, self.ncols), dtype=np.float64)
                counts[y][m] = np.zeros((self.nrows, self.ncols), dtype=np.int32)

            valid = ~np.isnan(data)
            sums[y][m][valid] += data[valid]
            counts[y][m][valid] += 1

        means = {}
        for y in sums:
            means[y] = {}
            for m in sums[y]:
                means[y][m] = np.where(
                    counts[y][m] > 0,
                    sums[y][m] / counts[y][m],
                    np.nan
                ).astype(np.float32)

        return means

    def compute_bias_factors(self, cal_means, src_means, source_name):
        """
        Compute bias factors for a single source.

        bias = M_sta - M_src

        Parameters
        ----------
        cal_means : dict
            Calibration monthly means {year: {month: array}}.
        src_means : dict
            Source monthly means {year: {month: array}}.
        source_name : str
            Name of the source (for logging).

        Returns
        -------
        dict
            Bias factors {year: {month: array}}.
        """
        if src_means is None:
            return {}

        bias = {}

        for y in cal_means:
            if y not in src_means:
                continue

            bias[y] = {}

            for m in cal_means[y]:
                if m not in src_means[y]:
                    continue

                b = cal_means[y][m] - src_means[y][m]
                b = np.clip(b, -self.correction_limit, self.correction_limit)
                bias[y][m] = b.astype(np.float32)

        return bias

    def apply_correction(self, data, date_obj, source_name):
        """
        Apply monthly bias correction to a single day of data [Eq.(2)].

        Parameters
        ----------
        data : ndarray
            Input data to correct.
        date_obj : datetime
            Date of the data.
        source_name : str
            Name of the source product.

        Returns
        -------
        ndarray
            Bias-corrected data.
        """
        if self.bias_factors is None or data is None:
            return data

        y, m = date_obj.year, date_obj.month

        if source_name not in self.bias_factors:
            return data
        if y not in self.bias_factors[source_name]:
            return data
        if m not in self.bias_factors[source_name][y]:
            return data

        bias = self.bias_factors[source_name][y][m]
        if bias.shape == data.shape:
            data = data + bias

        return data
