"""
I/O utilities for reading/writing GeoTIFF and NetCDF files.
"""

import os
import glob
import re
from datetime import datetime
import numpy as np
from osgeo import gdal
import netCDF4 as nc


def extract_date_from_filename(filename):
    """
    Extract 8-digit date string (YYYYMMDD) from a filename using pattern matching.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    patterns = [
        r'(\d{4})(\d{2})(\d{2})',
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{4})_(\d{2})_(\d{2})',
        r'(\d{4})\.(\d{2})\.(\d{2})'
    ]

    for pattern in patterns:
        m = re.search(pattern, basename)
        if m:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return None


def build_file_map(data_dir):
    """
    Build a dictionary mapping dates to file paths for all TIFFs in a directory.
    """
    fm = {}
    if not os.path.exists(data_dir):
        return fm

    for ext in ['*.tif', '*.tiff']:
        for f in glob.glob(os.path.join(data_dir, ext)):
            date_str = extract_date_from_filename(f)
            if date_str:
                try:
                    dt = datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                    fm[dt] = f
                except ValueError:
                    continue

    return fm


def read_raster(file_path):
    """
    Read a GeoTIFF file and return the data array and no-data value.
    """
    ds = gdal.Open(file_path)
    if ds is None:
        return None, None

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    data = ds.ReadAsArray().astype(np.float32)
    ds = None

    if nodata is not None:
        data[data == nodata] = np.nan

    return data, nodata


def write_raster(file_path, data, geotransform, projection, nodata=-9999):
    """
    Write a 2D array to a GeoTIFF file with the given georeferencing.
    """
    if data is None:
        return

    nrows, ncols = data.shape

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(file_path, ncols, nrows, 1, gdal.GDT_Float32,
                       ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    out_data = data.copy()
    out_data[np.isnan(out_data)] = nodata
    ds.GetRasterBand(1).WriteArray(out_data)
    ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds = None


def get_common_dates(maps):
    """
    Find dates that are present in all provided file maps.
    """
    if len(maps) == 0:
        return []

    common = set(maps[0].keys())
    for m in maps[1:]:
        common = common & set(m.keys())

    return sorted(common)


def read_netcdf_hourly(file_path, variable_name):
    """
    Read hourly data from a NetCDF file and return the data array.
    """
    if not os.path.exists(file_path):
        return None

    try:
        ds = nc.Dataset(file_path, 'r')
        if variable_name not in ds.variables:
            ds.close()
            return None

        data = ds.variables[variable_name][:]
        ds.close()
        return data
    except Exception:
        return None


def write_netcdf_hourly(file_path, data, lats, lons, time_units,
                        variable_name, long_name, units):
    """
    Write hourly data to a NetCDF file following CF conventions.
    """
    ntimes, nlat, nlon = data.shape

    ds = nc.Dataset(file_path, 'w', format='NETCDF4')

    # Create dimensions
    ds.createDimension('time', ntimes)
    ds.createDimension('lat', nlat)
    ds.createDimension('lon', nlon)

    # Create coordinate variables
    time_var = ds.createVariable('time', 'f8', ('time',))
    time_var.units = time_units
    time_var.standard_name = 'time'

    lat_var = ds.createVariable('lat', 'f8', ('lat',))
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'
    lat_var[:] = lats

    lon_var = ds.createVariable('lon', 'f8', ('lon',))
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'
    lon_var[:] = lons

    # Create data variable
    data_var = ds.createVariable(variable_name, 'f4', ('time', 'lat', 'lon'),
                                 zlib=True, complevel=4)
    data_var.units = units
    data_var.long_name = long_name
    data_var.standard_name = variable_name
    data_var.missing_value = -9999.0

    # Write data
    out_data = data.copy()
    out_data[np.isnan(out_data)] = -9999.0
    data_var[:] = out_data

    # Add global attributes
    ds.description = 'Hourly downscaled meteorological data from EIVD fusion'

    ds.close()
