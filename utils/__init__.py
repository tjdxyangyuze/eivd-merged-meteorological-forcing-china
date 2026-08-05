# This package provides I/O, grid manipulation, and validation utilities.

from .io_utils import (
    extract_date_from_filename,
    build_file_map,
    read_raster,
    write_raster,
    get_common_dates,
    read_netcdf_hourly,
    write_netcdf_hourly
)
from .grid_utils import (
    get_pixel_coords,
    compute_geographic_bounds,
    create_coordinate_grids
)
from .validation import (
    rmse,
    mae,
    bias,
    compute_metrics,
    pod,
    far,
    csi
)
