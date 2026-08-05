"""
Grid manipulation utilities for coordinate transformation and grid operations.
"""

import numpy as np


def get_pixel_coords(lon, lat, geotransform, nrows, ncols):
    """
    Convert geographic coordinates to pixel indices.

    Returns
    -------
    tuple or (None, None)
        (row, col) pixel indices, or (None, None) if coordinates are out of bounds.
    """
    resolution = abs(geotransform[1])
    lon_min = geotransform[0]
    lat_max = geotransform[3]

    col = int((lon - lon_min) / resolution + 0.5)
    row = int((lat_max - lat) / resolution + 0.5)

    if col < 0 or col >= ncols or row < 0 or row >= nrows:
        return None, None

    return row, col


def compute_geographic_bounds(geotransform, nrows, ncols):
    """
    Compute geographic bounds from a GDAL geotransform.

    Returns
    -------
    dict
        Dictionary containing lon_min, lon_max, lat_min, lat_max, and resolution.
    """
    resolution = abs(geotransform[1])
    lon_min = geotransform[0]
    lon_max = geotransform[0] + ncols * resolution
    lat_max = geotransform[3]
    lat_min = geotransform[3] + nrows * resolution

    return {
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'resolution': resolution
    }


def create_coordinate_grids(geotransform, nrows, ncols):
    """
    Create 2D longitude and latitude grids.

    Returns
    -------
    tuple
        (lon_grid, lat_grid) where each is a 2D array of shape (nrows, ncols).
    """
    resolution = abs(geotransform[1])
    lon_min = geotransform[0]
    lat_max = geotransform[3]

    lon_1d = np.arange(lon_min + resolution/2, lon_min + ncols * resolution, resolution)
    lat_1d = np.arange(lat_max - resolution/2, lat_max - nrows * resolution, -resolution)

    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    return lon_grid, lat_grid
