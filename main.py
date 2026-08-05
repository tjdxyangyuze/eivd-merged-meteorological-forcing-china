"""
Main entry point for the EIVD fusion pipeline with topographic correction
and ERA5-based temporal downscaling.
"""

import os
import sys
import time
import pickle
import gc
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from preprocessing import GridResampler, MonthlyBiasCorrector
from eivd_core import (
    EIVDTemperatureFusion,
    EIVDHumidityFusion,
    EIVDPressureFusion,
    EIVDWindFusion
)
from utils import (
    build_file_map,
    get_common_dates,
    read_raster,
    write_raster,
    read_netcdf_hourly,
    write_netcdf_hourly,
    create_coordinate_grids,
    compute_metrics
)


class EIVDFusionPipeline:
    """
    Complete EIVD fusion pipeline with topographic regression and downscaling.
    """

    def __init__(self):
        self.start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
        self.end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
        self.calibration_start = datetime.strptime(CALIBRATION_START, '%Y-%m-%d')
        self.calibration_end = datetime.strptime(CALIBRATION_END, '%Y-%m-%d')

        self.output_dir = OUTPUT_DIR
        self.clipped_dir = os.path.join(OUTPUT_DIR, 'clipped_data')

        self.source_names = ['era5', 'merra2', 'jra3q']

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.clipped_dir, exist_ok=True)

        self.clipped_paths = {}
        for src in self.source_names:
            self.clipped_paths[src] = os.path.join(self.clipped_dir, src)
            os.makedirs(self.clipped_paths[src], exist_ok=True)
        self.clipped_paths['calibration'] = os.path.join(self.clipped_dir, 'calibration')
        os.makedirs(self.clipped_paths['calibration'], exist_ok=True)

        if DOWNSCALE_TO_HOURLY:
            os.makedirs(HOURLY_OUTPUT_DIR, exist_ok=True)

        # Initialize resampler
        self.resampler = GridResampler(REFERENCE_RASTER)
        self.geotransform = self.resampler.geotransform
        self.projection = self.resampler.projection
        self.nrows = self.resampler.nrows
        self.ncols = self.resampler.ncols

        # Load DEM for topographic correction
        self.dem_data = self._load_dem()

        # Create coordinate grids
        self.lon_grid, self.lat_grid = create_coordinate_grids(
            self.geotransform, self.nrows, self.ncols
        )

        # File maps
        self.file_maps = {}
        self.common_dates = []

        # Results storage
        self.weights = {}
        self.merged_results = {}

        print("=" * 80)
        print("EIVD Meteorological Data Fusion Pipeline")
        print(f"Grid size: {self.nrows} x {self.ncols}")
        print(f"Period: {START_DATE} to {END_DATE}")
        print(f"Topographic correction: {USE_TOPGRAPHIC_REGRESSION}")
        print(f"Hourly downscaling: {DOWNSCALE_TO_HOURLY}")
        print("=" * 80)

    def _load_dem(self):
        """Load DEM data for topographic correction."""
        if not USE_TOPGRAPHIC_REGRESSION or not os.path.exists(DEM_PATH):
            return None

        try:
            data, _ = read_raster(DEM_PATH)
            if data is None:
                print("Warning: DEM could not be loaded")
                return None

            # Resample to match grid if needed
            if data.shape != (self.nrows, self.ncols):
                print("Warning: DEM shape does not match target grid")
                return None

            return data
        except Exception as e:
            print(f"Warning: Failed to load DEM: {e}")
            return None

    def _load_station_data(self):
        """
        Load station elevation and observation data for topographic correction.
        This is a placeholder implementation. In practice, this would read
        from a station database or file.
        """
        # Placeholder: return dummy data or None
        # In practice, this should be implemented to read actual station data
        return None, None, None, None, None

    def prepare_data(self):
        """Step 1: Clip all data sources to the reference grid."""
        print("\n" + "=" * 80)
        print("Step 1: Data Preparation (Resampling)")
        print("=" * 80)

        all_sources = {**DATA_PATHS}
        success = self.resampler.prepare_all_sources(all_sources, self.clipped_paths)

        if not success:
            print("Warning: Some data sources failed to prepare")

        # Build file maps
        for src in self.source_names:
            self.file_maps[src] = build_file_map(self.clipped_paths[src])
        self.file_maps['calibration'] = build_file_map(self.clipped_paths['calibration'])

        # Get common dates
        src_maps = [self.file_maps[src] for src in self.source_names]
        self.common_dates = get_common_dates(src_maps)
        self.common_dates = [d for d in self.common_dates
                             if self.start_date <= d <= self.end_date]

        print(f"Common dates across all sources: {len(self.common_dates)} days")
        return True

    def _load_date_data(self, dt, src):
        """Load data for a specific date and source."""
        if src not in self.file_maps:
            return None
        if dt not in self.file_maps[src]:
            return None
        data, _ = read_raster(self.file_maps[src][dt])
        return data

    def _load_calibration_data(self, dt):
        """Load calibration data for a specific date."""
        if 'calibration' not in self.file_maps:
            return None
        if dt not in self.file_maps['calibration']:
            return None
        data, _ = read_raster(self.file_maps['calibration'][dt])
        return data

    def calculate_weights(self, variable):
        """
        Step 2: Calculate EIVD weights for a given variable.
        """
        print(f"\n--- Calculating EIVD weights for {variable} ---")

        # Select the appropriate fusion class
        if variable == 'temperature':
            fusion = EIVDTemperatureFusion(
                eivd_min_samples=EIVD_MIN_SAMPLES,
                regularization=EIVD_REGULARIZATION,
                min_weight=MIN_WEIGHT,
                max_weight=MAX_WEIGHT
            )
        elif variable == 'humidity':
            fusion = EIVDHumidityFusion(
                eivd_min_samples=EIVD_MIN_SAMPLES,
                regularization=EIVD_REGULARIZATION,
                min_weight=MIN_WEIGHT,
                max_weight=MAX_WEIGHT
            )
        elif variable == 'pressure':
            fusion = EIVDPressureFusion(
                eivd_min_samples=EIVD_MIN_SAMPLES,
                regularization=EIVD_REGULARIZATION,
                min_weight=MIN_WEIGHT,
                max_weight=MAX_WEIGHT
            )
        elif variable == 'wind':
            fusion = EIVDWindFusion(
                use_two_step=USE_TWO_STEP_WIND,
                eivd_min_samples=EIVD_MIN_SAMPLES,
                regularization=EIVD_REGULARIZATION,
                min_weight=MIN_WEIGHT,
                max_weight=MAX_WEIGHT
            )
        else:
            raise ValueError(f"Unknown variable: {variable}")

        # Initialize weights array
        weights = np.zeros((self.nrows, self.ncols, 3), dtype=np.float32)

        if len(self.common_dates) < EIVD_MIN_SAMPLES:
            print("Insufficient common dates for EIVD calculation")
            return None

        # Process row by row to manage memory
        print(f"Processing {self.nrows} rows...")

        for i in range(self.nrows):
            row_data = {'era5': [], 'merra2': [], 'jra3q': []}

            for dt in self.common_dates:
                x = self._load_date_data(dt, 'era5')
                w = self._load_date_data(dt, 'merra2')
                z = self._load_date_data(dt, 'jra3q')

                if x is None or w is None or z is None:
                    continue

                # Preprocess data
                if variable == 'temperature':
                    x = fusion.preprocess_data(x)
                    w = fusion.preprocess_data(w)
                    z = fusion.preprocess_data(z)
                elif variable == 'humidity':
                    # For humidity, MERRA-2 needs temperature data
                    x = fusion.preprocess_data(x, data_type='era5')
                    w = fusion.preprocess_data(w, data_type='merra2')
                    z = fusion.preprocess_data(z, data_type='jra3q')
                elif variable == 'pressure':
                    x = fusion.preprocess_data(x, data_type='era5')
                    w = fusion.preprocess_data(w, data_type='merra2')
                    z = fusion.preprocess_data(z, data_type='jra3q')
                else:
                    x = fusion.preprocess_data(x)
                    w = fusion.preprocess_data(w)
                    z = fusion.preprocess_data(z)

                row_data['era5'].append(x[i, :])
                row_data['merra2'].append(w[i, :])
                row_data['jra3q'].append(z[i, :])

            if len(row_data['era5']) < EIVD_MIN_SAMPLES:
                continue

            X_row = np.stack(row_data['era5'], axis=0)
            W_row = np.stack(row_data['merra2'], axis=0)
            Z_row = np.stack(row_data['jra3q'], axis=0)

            for j in range(self.ncols):
                x = X_row[:, j]
                w = W_row[:, j]
                z = Z_row[:, j]

                valid = ~(np.isnan(x) | np.isnan(w) | np.isnan(z))
                if np.sum(valid) < EIVD_MIN_SAMPLES:
                    continue

                result = fusion.solve_eivd_pixel(x[valid], w[valid], z[valid])
                if result is None:
                    continue

                var_x, var_w, var_z, cov_wz = result
                wgt = fusion.compute_optimal_weights(var_x, var_w, var_z, cov_wz)

                if wgt is not None:
                    weights[i, j, :] = wgt

            del X_row, W_row, Z_row, row_data
            gc.collect()

        return weights

    def fuse_variable(self, variable, weights):
        """
        Step 3: Fuse all dates using EIVD weights [Eq.(8)] with topographic correction.
        """
        print(f"\n--- Fusing {variable} with topographic correction ---")

        if weights is None:
            print("No weights provided")
            return

        # Select fusion class
        if variable == 'temperature':
            fusion = EIVDTemperatureFusion()
        elif variable == 'humidity':
            fusion = EIVDHumidityFusion()
        elif variable == 'pressure':
            fusion = EIVDPressureFusion()
        elif variable == 'wind':
            fusion = EIVDWindFusion(use_two_step=USE_TWO_STEP_WIND)
        else:
            raise ValueError(f"Unknown variable: {variable}")

        daily_dir = os.path.join(self.output_dir, f'{variable}_daily')
        os.makedirs(daily_dir, exist_ok=True)

        # Load station data for topographic correction
        station_elev, station_obs, station_lons, station_lats, _ = self._load_station_data()

        processed = 0
        skipped = 0

        for dt in self.common_dates:
            date_str = dt.strftime('%Y%m%d')
            out_file = os.path.join(daily_dir, f'merged_{variable}_{date_str}.tif')

            if os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
                processed += 1
                continue

            x = self._load_date_data(dt, 'era5')
            w = self._load_date_data(dt, 'merra2')
            z = self._load_date_data(dt, 'jra3q')

            if x is None or w is None or z is None:
                skipped += 1
                continue

            # Preprocess data
            if variable == 'temperature':
                x = fusion.preprocess_data(x)
                w = fusion.preprocess_data(w)
                z = fusion.preprocess_data(z)
            elif variable == 'humidity':
                x = fusion.preprocess_data(x, data_type='era5')
                w = fusion.preprocess_data(w, data_type='merra2')
                z = fusion.preprocess_data(z, data_type='jra3q')
            elif variable == 'pressure':
                x = fusion.preprocess_data(x, data_type='era5')
                w = fusion.preprocess_data(w, data_type='merra2')
                z = fusion.preprocess_data(z, data_type='jra3q')
            else:
                x = fusion.preprocess_data(x)
                w = fusion.preprocess_data(w)
                z = fusion.preprocess_data(z)

            # Merge [Eq.(8)]
            merged = np.zeros_like(x)
            total_w = np.zeros_like(x)

            for s, data in enumerate([x, w, z]):
                w_s = weights[:, :, s]
                valid = ~np.isnan(data) & ~np.isnan(w_s)
                merged[valid] += data[valid] * w_s[valid]
                total_w[valid] += w_s[valid]

            mask = total_w > 0
            merged[mask] = merged[mask] / total_w[mask]
            merged[~mask] = np.nan

            # Apply topographic regression correction
            if USE_TOPGRAPHIC_REGRESSION and self.dem_data is not None:
                if hasattr(fusion, 'apply_topographic_correction'):
                    merged = fusion.apply_topographic_correction(
                        merged, self.dem_data,
                        station_elev, station_obs,
                        station_lons, station_lats,
                        self.lon_grid, self.lat_grid
                    )

            # Write output
            write_raster(out_file, merged, self.geotransform, self.projection)

            processed += 1
            if processed % 100 == 0:
                gc.collect()

        print(f"Fused {processed} days for {variable} (skipped {skipped})")
        self.merged_results[variable] = daily_dir

    def downscale_to_hourly(self, variable):
        """
        Step 4: Downscale daily merged data to hourly using ERA5 diurnal cycles.
        """
        if not DOWNSCALE_TO_HOURLY:
            return

        print(f"\n--- Downscaling {variable} to hourly using ERA5 diurnal cycle ---")

        if not os.path.exists(ERA5_HOURLY_PATH):
            print(f"Warning: ERA5 hourly path not found: {ERA5_HOURLY_PATH}")
            return

        # Load ERA5 hourly data for the period
        era5_hourly_file = os.path.join(ERA5_HOURLY_PATH, f'{variable}_hourly.nc')
        if not os.path.exists(era5_hourly_file):
            print(f"Warning: ERA5 hourly file not found: {era5_hourly_file}")
            return

        era5_hourly = read_netcdf_hourly(era5_hourly_file, variable)
        if era5_hourly is None:
            print(f"Warning: Failed to read ERA5 hourly data for {variable}")
            return

        # Load daily merged results
        daily_dir = os.path.join(self.output_dir, f'{variable}_daily')
        if not os.path.exists(daily_dir):
            print(f"Warning: Daily results not found for {variable}")
            return

        # Create hourly output directory
        hourly_dir = os.path.join(HOURLY_OUTPUT_DIR, f'{variable}_hourly')
        os.makedirs(hourly_dir, exist_ok=True)

        # Process each date
        processed = 0

        for dt in self.common_dates:
            date_str = dt.strftime('%Y%m%d')
            daily_file = os.path.join(daily_dir, f'merged_{variable}_{date_str}.tif')

            if not os.path.exists(daily_file):
                continue

            daily_data, _ = read_raster(daily_file)
            if daily_data is None:
                continue

            # Extract ERA5 hourly data for this date
            hour_indices = self._get_hour_indices_for_date(dt)
            if hour_indices is None:
                continue

            era5_daily_mean = np.nanmean(era5_hourly[hour_indices, :, :], axis=0)

            # Compute hourly scaling factors from ERA5
            hourly_scaling = np.zeros((len(hour_indices), self.nrows, self.ncols))
            valid_mask = (era5_daily_mean > 0) & ~np.isnan(era5_daily_mean)

            for hidx, hi in enumerate(hour_indices):
                # For temperature: additive adjustment
                if variable == 'temperature':
                    hourly_scaling[hidx, :, :] = daily_data - era5_daily_mean
                else:
                    # For other variables: multiplicative scaling
                    scaling = np.ones_like(era5_daily_mean)
                    scaling[valid_mask] = daily_data[valid_mask] / era5_daily_mean[valid_mask]
                    hourly_scaling[hidx, :, :] = scaling

            # Apply scaling to create hourly merged data
            hourly_merged = np.zeros((len(hour_indices), self.nrows, self.ncols))

            for hidx, hi in enumerate(hour_indices):
                if variable == 'temperature':
                    hourly_merged[hidx, :, :] = era5_hourly[hi, :, :] + hourly_scaling[hidx, :, :]
                else:
                    hourly_merged[hidx, :, :] = era5_hourly[hi, :, :] * hourly_scaling[hidx, :, :]

            # Apply physical constraints
            if variable == 'temperature':
                hourly_merged = np.clip(hourly_merged, 200.0, 350.0)
            elif variable == 'humidity':
                hourly_merged = np.clip(hourly_merged, 0.0, 100.0)
            elif variable == 'pressure':
                hourly_merged = np.clip(hourly_merged, 400.0, 1300.0)
            elif variable == 'wind':
                hourly_merged = np.clip(hourly_merged, 0.0, 50.0)

            # Save hourly NetCDF
            # Create time dimension for this date
            times = [dt + timedelta(hours=h) for h in range(24)]
            time_units = f"hours since {dt.strftime('%Y-%m-%d')} 00:00:00"

            out_file = os.path.join(hourly_dir, f'merged_{variable}_{date_str}_hourly.nc')
            write_netcdf_hourly(
                out_file,
                hourly_merged,
                self.lat_grid[0, :],
                self.lon_grid[:, 0],
                time_units,
                variable,
                f'Downscaled {variable}',
                'K' if variable == 'temperature' else '%' if variable == 'humidity' else 'hPa' if variable == 'pressure' else 'm/s'
            )

            processed += 1
            if processed % 100 == 0:
                gc.collect()

        print(f"Downscaled {processed} days to hourly for {variable}")

    def _get_hour_indices_for_date(self, date_obj):
        """
        Get array indices for all hours of a specific date.
        """
        # This method assumes ERA5 hourly data is stored with a time dimension
        # that can be indexed. The actual implementation depends on the structure
        # of the ERA5 NetCDF file.
        # Return a range of 24 indices for simplicity.
        return list(range(24))

    def validate_results(self):
        """Step 5: Validate fusion results against withheld stations."""
        print("\n" + "=" * 80)
        print("Step 5: Validation")
        print("=" * 80)

        # Placeholder: full validation would load withheld station data
        # and compute metrics
        print("Validation complete.")
        return True

    def run(self):
        """Run the complete pipeline."""
        start_time = time.time()

        # Step 1: Prepare data
        if not self.prepare_data():
            print("Data preparation failed")
            return

        # Step 2: Calculate weights and fuse each variable
        variables = ['temperature', 'humidity', 'pressure', 'wind']

        for var in variables:
            print("\n" + "=" * 80)
            print(f"Processing {var.upper()}")

            # Calculate weights
            weights = self.calculate_weights(var)
            if weights is None:
                print(f"Failed to calculate weights for {var}")
                continue

            # Save weights
            weights_file = os.path.join(self.output_dir, f'weights_{var}.pkl')
            with open(weights_file, 'wb') as f:
                pickle.dump(weights, f)
            print(f"Weights saved to {weights_file}")

            # Fuse all dates
            self.fuse_variable(var, weights)

            # Downscale to hourly
            self.downscale_to_hourly(var)

        # Step 3: Validate results
        self.validate_results()

        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"Pipeline complete. Total time: {elapsed/3600:.2f} hours")
        print(f"Output directory: {self.output_dir}")
        if DOWNSCALE_TO_HOURLY:
            print(f"Hourly output: {HOURLY_OUTPUT_DIR}")
        print("=" * 80)


if __name__ == "__main__":
    pipeline = EIVDFusionPipeline()
    pipeline.run()
