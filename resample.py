"""
Spatial resampling module.

This module handles clipping and resampling of input data to the target
reference grid using GDAL Warp functionality.
"""

import os
import glob
from osgeo import gdal
from tqdm import tqdm


class GridResampler:
    """
    Resample data to a target reference grid.

    This class handles the spatial resampling of all input data sources
    to match the target grid defined by a reference raster.
    """

    def __init__(self, reference_raster_path):
        """
        Initialize the resampler with a reference grid.

        Parameters
        ----------
        reference_raster_path : str
            Path to the reference raster defining target grid
        """
        self.reference_raster_path = reference_raster_path
        self._load_reference_info()

    def _load_reference_info(self):
        """Extract georeferencing information from reference raster."""
        ds = gdal.Open(self.reference_raster_path)
        if ds is None:
            raise ValueError("Unable to open reference raster")

        self.ncols = ds.RasterXSize
        self.nrows = ds.RasterYSize
        self.geotransform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()
        self.lon_min = self.geotransform[0]
        self.lon_max = self.geotransform[0] + self.ncols * self.geotransform[1]
        self.lat_max = self.geotransform[3]
        self.lat_min = self.geotransform[3] + self.nrows * self.geotransform[5]
        self.resolution = abs(self.geotransform[1])
        ds = None

    def clip_to_grid(self, input_dir, output_dir, force_reclip=False):
        """
        Clip all files in input_dir to the reference grid.

        Parameters
        ----------
        input_dir : str
            Directory containing input GeoTIFF files
        output_dir : str
            Directory for clipped output files
        force_reclip : bool
            If True, reclip all files even if already clipped

        Returns
        -------
        bool
            True if successful
        """
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory not found: {input_dir}")
            return False

        os.makedirs(output_dir, exist_ok=True)

        raw_files = glob.glob(os.path.join(input_dir, "*.tif"))
        if not raw_files:
            print(f"Warning: No TIFF files in {input_dir}")
            return False

        existing = glob.glob(os.path.join(output_dir, "*.tif"))
        if not force_reclip and len(existing) >= len(raw_files) * 0.9:
            print(f"{input_dir}: already clipped, skipping")
            return True

        print(f"Clipping {len(raw_files)} files from {input_dir}...")

        for f in tqdm(raw_files, desc="Clipping"):
            out_file = os.path.join(output_dir, os.path.basename(f))
            if os.path.exists(out_file) and not force_reclip:
                continue

            try:
                gdal.Warp(out_file, f, options=gdal.WarpOptions(
                    format='GTiff',
                    outputBounds=[self.lon_min, self.lat_min,
                                  self.lon_max, self.lat_max],
                    xRes=self.resolution,
                    yRes=self.resolution,
                    resampleAlg='bilinear',
                    dstSRS=self.projection,
                    outputType=gdal.GDT_Float32,
                    creationOptions=['COMPRESS=LZW', 'TILED=YES']
                ))
            except Exception as e:
                print(f"  Failed: {os.path.basename(f)} - {e}")

        print(f"Clipping complete: {output_dir}")
        return True

    def prepare_all_sources(self, source_paths, clipped_paths):
        """
        Prepare all data sources by clipping to reference grid.

        Parameters
        ----------
        source_paths : dict
            {source_name: input_directory}
        clipped_paths : dict
            {source_name: output_directory}

        Returns
        -------
        bool
            True if all sources prepared successfully
        """
        for name, in_path in source_paths.items():
            out_path = clipped_paths.get(name)
            if out_path is None:
                continue
            if not self.clip_to_grid(in_path, out_path):
                print(f"Warning: {name} preparation failed")
                return False
        return True
