"""
EIVDFusionBase: Core Extended Double Instrumental Variable algorithm [Eq.(4)-(8)].
"""

import numpy as np


class EIVDFusionBase:
    """
    Base class implementing the core EIVD algorithm.

    References:
        Dong, J., et al. (2020). J. Hydrol., 581, 124413.

    Key equations:
        Eq.(4): X_i = α_i * p + β_i + ε_i
        Eq.(5): Error covariance matrix with cross-correlation between w and z
        Eq.(6): Var(e_merged) = w^T * Σ * w
        Eq.(7): w* = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
        Eq.(8): X_merged = w_x* * X_corr_x + w_w* * X_corr_w + w_z* * X_corr_z
    """

    def __init__(self, eivd_min_samples=30, regularization=1e-4,
                 min_weight=-1.0, max_weight=1.0):
        """
        Initialize the EIVD fusion base.

        Parameters
        ----------
        eivd_min_samples : int
            Minimum number of samples required for EIVD estimation.
        regularization : float
            Regularization factor added to the covariance matrix diagonal.
        min_weight : float
            Minimum allowed weight value.
        max_weight : float
            Maximum allowed weight value.
        """
        self.eivd_min_samples = eivd_min_samples
        self.regularization = regularization
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weights = None

    def solve_eivd_pixel(self, x, w, z):
        """
        Solve the EIVD system for a single pixel [Eq.(4)-(5)].

        The method uses lag-1 auto-covariances as instrumental variables to
        estimate error variances and the error covariance between w and z.

        Parameters
        ----------
        x, w, z : ndarray
            Time series of three products (x=ERA5, w=MERRA2, z=JRA3Q).

        Returns
        -------
        tuple or None
            (var_x, var_w, var_z, cov_wz) or None if estimation fails.
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

        # Lag-1 auto-covariances (instrumental variables)
        if len(x) > 1:
            Lx = np.cov(x[:-1], x[1:], ddof=1)[0, 1]
            Lw = np.cov(w[:-1], w[1:], ddof=1)[0, 1]
            Lz = np.cov(z[:-1], z[1:], ddof=1)[0, 1]
        else:
            return None

        # Auto-covariances must be positive for valid instrumental variables
        if Lx <= 0 or Lw <= 0 or Lz <= 0:
            return None

        # Estimate scaling factors from lagged covariances
        ratio_xw = np.sqrt(max(Lx / Lw, 1e-6))
        ratio_xz = np.sqrt(max(Lx / Lz, 1e-6))

        # Check for sign consistency
        if Cxw * ratio_xw < 0 or Cxz * ratio_xz < 0:
            return None

        # Set reference scaling factor (ERA5 as reference)
        alpha_x = 1.0
        alpha_w = 1.0 / ratio_xw
        alpha_z = 1.0 / ratio_xz

        # Estimate truth variance Cpp from two independent estimates
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

        Parameters
        ----------
        var_x, var_w, var_z : float
            Error variances for the three products.
        cov_wz : float
            Error covariance between w and z.

        Returns
        -------
        ndarray or None
            Array of three optimal weights, or None if inversion fails.
        """
        # Build error covariance matrix [Eq.(5)]
        Sigma = np.array([
            [var_x, 0.0, 0.0],
            [0.0, var_w, cov_wz],
            [0.0, cov_wz, var_z]
        ])

        # Regularization to prevent singular matrix
        Sigma += np.eye(3) * self.regularization

        try:
            inv_S = np.linalg.inv(Sigma)
            ones = np.ones(3)
            # Eq.(7): w* = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
            wgt = inv_S @ ones / (ones @ inv_S @ ones)
            wgt = np.clip(wgt, self.min_weight, self.max_weight)

            if np.sum(wgt) > 0:
                wgt = wgt / np.sum(wgt)
                return wgt
            return None

        except np.linalg.LinAlgError:
            return None

    def merge_pixel(self, x, w, z, weights):
        """
        Merge three products using optimal weights [Eq.(8)].

        Parameters
        ----------
        x, w, z : float or ndarray
            Corrected product values.
        weights : ndarray
            Array of three optimal weights.

        Returns
        -------
        float or ndarray
            Merged value.
        """
        if weights is None:
            return np.nan
        return weights[0] * x + weights[1] * w + weights[2] * z

    def apply_topographic_correction(self, merged_data, dem_data, station_elev,
                                     station_obs, station_lons, station_lats,
                                     lon_grid, lat_grid):
        """
        Apply topographic regression correction using station residuals [Section 4].

        The method regresses station residuals (observation - merged value)
        against station elevation and applies the resulting slope to all grid cells.

        Parameters
        ----------
        merged_data : ndarray
            Merged field to be corrected (2D).
        dem_data : ndarray
            Digital elevation model (2D).
        station_elev : ndarray
            Elevation values at station locations (1D).
        station_obs : ndarray
            Observed values at station locations (1D).
        station_lons, station_lats : ndarray
            Station coordinates.
        lon_grid, lat_grid : ndarray
            Grid coordinate arrays.

        Returns
        -------
        ndarray
            Topographically-corrected merged field.
        """
        if dem_data is None or station_elev is None or len(station_elev) < 3:
            return merged_data

        # Extract merged values at station locations using nearest-neighbor
        merged_at_stations = []
        valid_station_indices = []

        for idx, (slon, slat) in enumerate(zip(station_lons, station_lats)):
            # Find nearest grid cell
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

        # Compute residuals: residual = observation - merged_value
        residuals = station_obs_valid - merged_at_stations

        # Perform linear regression: residual ~ elevation
        # Remove outliers using 2-sigma threshold
        valid_mask = ~np.isnan(residuals) & ~np.isnan(station_elev_valid)
        if np.sum(valid_mask) < 3:
            return merged_data

        x_reg = station_elev_valid[valid_mask]
        y_reg = residuals[valid_mask]

        # Simple linear regression
        x_mean = np.mean(x_reg)
        y_mean = np.mean(y_reg)
        numerator = np.sum((x_reg - x_mean) * (y_reg - y_mean))
        denominator = np.sum((x_reg - x_mean) ** 2)

        if denominator > 0:
            slope = numerator / denominator
            # Limit slope to prevent over-correction
            slope = np.clip(slope, -self.correction_limit, self.correction_limit)
            intercept = y_mean - slope * x_mean

            # Apply correction to all grid cells
            corrected = merged_data + slope * (dem_data - np.mean(x_reg))
            return np.clip(corrected, merged_data - 10, merged_data + 10)

        return merged_data
