"""
User configuration for the EIVD fusion framework.
"""

# ============================================================
# Time Range Configuration
# ============================================================

START_DATE = '1980-01-01'
END_DATE = '2020-12-31'
CALIBRATION_START = '1980-01-01'
CALIBRATION_END = '2020-12-31'

# ============================================================
# Data Path Configuration
# ============================================================

REFERENCE_RASTER = '/path/to/reference_grid.tif'

DATA_PATHS = {
    'era5': '/path/to/era5_data',
    'merra2': '/path/to/merra2_data',
    'jra3q': '/path/to/jra3q_data',
    'calibration': '/path/to/cma_station_interpolated'
}

# ERA5 hourly data for downscaling (separate directory)
ERA5_HOURLY_PATH = '/path/to/era5_hourly_data'

# DEM file for topographic regression
DEM_PATH = '/path/to/dem_0.1deg.tif'

# Station U/V data for wind decomposition (optional)
STATION_UV_PATH = '/path/to/station_uv_data'

OUTPUT_DIR = '/path/to/output_directory'

# ============================================================
# EIVD Parameters
# ============================================================

EIVD_MIN_SAMPLES = 30
EIVD_REGULARIZATION = 1e-4
MIN_WEIGHT = -1.0
MAX_WEIGHT = 1.0

# ============================================================
# Topographic Regression Parameters
# ============================================================

TOPOGRAPHIC_REGRESSION_LAPSE_RATE_LIMIT = 0.01

# ============================================================
# Time Downscaling Parameters
# ============================================================

DOWNSCALE_TO_HOURLY = True
HOURLY_OUTPUT_DIR = '/path/to/hourly_output'

# ============================================================
# Flags
# ============================================================

USE_TOPGRAPHIC_REGRESSION = True
USE_TWO_STEP_WIND = True
VALIDATE_RESULTS = True
