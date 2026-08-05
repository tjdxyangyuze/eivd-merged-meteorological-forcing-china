import numpy as np
from tqdm import tqdm


class MonthlyBiasCorrector:
    """
    Compute and apply monthly bias correction [Eq.(2)].
    bias = M_sta - M_src
    X_corr = X + bias
    """

    def __init__(self, start_date, end_date,
                 calibration_start, calibration_end,
                 nrows, ncols,
                 correction_limit=50.0,
                 nodata_values=None):

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
