"""
StationInterpolator: Interpolate discrete station observations to a regular grid
using Delaunay triangulation-based linear interpolation with nearest-neighbor
extrapolation.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from tqdm import tqdm


class StationInterpolator:
    """
    Interpolate station observations to a regular grid.

    Method:
        1. Delaunay triangulation-based linear interpolation inside the convex hull.
        2. Nearest-neighbor extrapolation outside the convex hull.
    """

    def __init__(self, method='tri_linear_nearest'):
        """
        Parameters
        ----------
        method : str
            Interpolation method. Supports 'tri_linear_nearest'.
        """
        self.method = method

    def interpolate_tri_linear_nearest(self, station_lons, station_lats,
                                       station_values, target_lons, target_lats):
        """
        Interpolate using Delaunay linear interpolation inside the convex hull
        and nearest-neighbor extrapolation outside.

        Parameters
        ----------
        station_lons, station_lats : ndarray
            Station coordinates (1D).
        station_values : ndarray
            Station observations (1D).
        target_lons, target_lats : ndarray
            Target grid coordinates (2D).

        Returns
        -------
        ndarray
            Interpolated field on the target grid (2D).
        """
        # Remove invalid stations
        valid = ~np.isnan(station_values)
        slon = station_lons[valid]
        slat = station_lats[valid]
        sval = station_values[valid]

        if len(sval) < 3:
            return np.full_like(target_lons, np.nan, dtype=np.float32)

        # Station points and flattened target coordinates
        points = np.column_stack([slon, slat])
        target_shape = target_lons.shape
        target = np.column_stack([target_lons.ravel(), target_lats.ravel()])

        # Step 1: Delaunay triangulation-based linear interpolation
        # Returns NaN for points outside the convex hull
        try:
            linear_interp = LinearNDInterpolator(points, sval)
            result = linear_interp(target)
        except Exception as e:
            print("  Triangulation failed: " + str(e))
            return np.full_like(target_lons, np.nan, dtype=np.float32)

        # Step 2: Nearest-neighbor extrapolation for outside-hull points
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            try:
                nearest_interp = NearestNDInterpolator(points, sval)
                result[nan_mask] = nearest_interp(target[nan_mask])
            except Exception as e:
                print("  Nearest-neighbor fallback failed: " + str(e))

        return result.reshape(target_shape).astype(np.float32)

    def interpolate(self, station_lons, station_lats, station_values,
                    target_lons, target_lats):
        """Dispatch to the configured interpolation method."""
        if self.method == 'tri_linear_nearest':
            return self.interpolate_tri_linear_nearest(
                station_lons, station_lats, station_values,
                target_lons, target_lats
            )
        else:
            raise ValueError("Unknown interpolation method: " + self.method)

    def interpolate_monthly(self, station_file_pattern, target_lons, target_lats,
                            months, variable_name='value'):
        """
        Interpolate monthly station data to grid for a list of months.

        Parameters
        ----------
        station_file_pattern : str
            File pattern for monthly station data (CSV: lon, lat, value).
        target_lons, target_lats : ndarray
            Target grid coordinates (2D).
        months : list of (year, month) tuples
            List of months to process.
        variable_name : str
            Column name for station values.

        Returns
        -------
        dict
            {year: {month: 2D array}}
        """
        results = {}

        for year, month in tqdm(months, desc="Interpolating stations"):
            station_file = station_file_pattern.format(year=year, month=month)
            if not os.path.exists(station_file):
                print("  Warning: Station file not found: " + station_file)
                continue

            try:
                df = pd.read_csv(station_file)
                slon = df['lon'].values
                slat = df['lat'].values
                sval = df[variable_name].values
            except Exception as e:
                print("  Failed to read " + station_file + ": " + str(e))
                continue

            grid = self.interpolate(slon, slat, sval, target_lons, target_lats)

            if year not in results:
                results[year] = {}
            results[year][month] = grid

        return results
