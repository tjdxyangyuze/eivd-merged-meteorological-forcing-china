import numpy as np


class EIVDFusionBase:
    """
    Core EIVD algorithm implementation.

    References:
        Dong, J., et al. (2020). J. Hydrol., 581, 124413.
    """

    def __init__(self, eivd_min_samples=30, regularization=1e-4,
                 min_weight=-1.0, max_weight=1.0):

        self.eivd_min_samples = eivd_min_samples
        self.regularization = regularization
        self.min_weight = min_weight
        self.max_weight = max_weight

    def solve_eivd_pixel(self, x, w, z):
        """
        Solve EIVD for a single pixel [Eq.(4)-(5)].

        Returns (var_x, var_w, var_z, cov_wz) or None.
        """
        if len(x) < self.eivd_min_samples:
            return None

        # Observation covariances
        Vx = np.var(x, ddof=1)
        Vw = np.var(w, ddof=1)
        Vz = np.var(z, ddof=1)
        Cxw = np.cov(x, w, ddof=1)[0, 1]
        Cxz = np.cov(x, z, ddof=1)[0, 1]
        Cwz = np.cov(w, z, ddof=1)[0, 1]

        # Lag-1 auto-covariances
        if len(x) > 1:
            Lx = np.cov(x[:-1], x[1:], ddof=1)[0, 1]
            Lw = np.cov(w[:-1], w[1:], ddof=1)[0, 1]
            Lz = np.cov(z[:-1], z[1:], ddof=1)[0, 1]
        else:
            return None

        if Lx <= 0 or Lw <= 0 or Lz <= 0:
            return None

        # Estimate scaling factors from lagged covariances
        ratio_xw = np.sqrt(max(Lx / Lw, 1e-6))
        ratio_xz = np.sqrt(max(Lx / Lz, 1e-6))

        if Cxw * ratio_xw < 0 or Cxz * ratio_xz < 0:
            return None

        alpha_x = 1.0
        alpha_w = 1.0 / ratio_xw
        alpha_z = 1.0 / ratio_xz

        # Estimate truth variance Cpp
        Cpp1 = Cxw / (alpha_x * alpha_w) if (alpha_x * alpha_w) != 0 else np.nan
        Cpp2 = Cxz / (alpha_x * alpha_z) if (alpha_x * alpha_z) != 0 else np.nan

        if np.isnan(Cpp1) or np.isnan(Cpp2) or Cpp1 <= 0 or Cpp2 <= 0:
            Cpp = max(Cpp1, Cpp2, 0.0)
        else:
            Cpp = (Cpp1 + Cpp2) / 2.0

        if Cpp <= 0:
            return None

        # Error variances [Eq.(4)]
        var_x = max(Vx - alpha_x**2 * Cpp, 0.0)
        var_w = max(Vw - alpha_w**2 * Cpp, 0.0)
        var_z = max(Vz - alpha_z**2 * Cpp, 0.0)

        # Error covariance between w and z [Eq.(5)]
        cov_wz = Cwz - alpha_w * alpha_z * Cpp

        if any(np.isnan([var_x, var_w, var_z, cov_wz])):
            return None

        return var_x, var_w, var_z, cov_wz

    def compute_optimal_weights(self, var_x, var_w, var_z, cov_wz):
        """
        Compute BLUE optimal weights [Eq.(6)-(7)].
        """
        Sigma = np.array([
            [var_x, 0.0, 0.0],
            [0.0, var_w, cov_wz],
            [0.0, cov_wz, var_z]
        ])

        Sigma += np.eye(3) * self.regularization

        try:
            inv_S = np.linalg.inv(Sigma)
            ones = np.ones(3)
            wgt = inv_S @ ones / (ones @ inv_S @ ones)
            wgt = np.clip(wgt, self.min_weight, self.max_weight)

            if np.sum(wgt) > 0:
                wgt = wgt / np.sum(wgt)
                return wgt
            else:
                return None

        except np.linalg.LinAlgError:
            return None

    def merge_pixel(self, x, w, z, weights):
        """
        Merge three products using optimal weights [Eq.(8)].
        """
        if weights is None:
            return np.nan

        merged = weights[0] * x + weights[1] * w + weights[2] * z
        return merged
