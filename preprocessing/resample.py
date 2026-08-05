"""
GridResampler: Clip and resample raster data to a target reference grid.
"""

import os
import glob
import numpy as np
from osgeo import gdal
from tqdm import tqdm


class GridResampler:
    """
    Resample any raster dataset to match the spatial grid of a reference file.
    All outputs are saved as GeoTIFF with consistent projection and extent.
    """

    def __init__(self, reference_raster_path):
        """
        Initialize the resampler with a reference grid.

        Parameters
        ----------
        reference_raster_path : str
            Path to the reference raster defining the target grid.
        """
        self.reference_raster_path = reference_raster_path
        self._load_reference_info()

    def _load_reference_info(self):
        """Extract georeferencing parameters from the reference raster."""
        ds = gdal.Open(self.reference_raster_path)
        if ds is None:
            raise ValueError("Unable to open reference raster: " + self.reference_raster_path)

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
        Clip all TIFF files in input_dir to the reference grid.

        Parameters
        ----------
        input_dir : str
            Directory containing input GeoTIFF files.
        output_dir : str
            Directory for clipped output files.
        force_reclip : bool
            If True, reprocess all files even if they already exist.

        Returns
        -------
        bool
            True if at least one file was successfully processed.
        """
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory not found: {input_dir}")
            return False

        os.makedirs(output_dir, exist_ok=True)

        raw_files = glob.glob(os.path.join(input_dir, "*.tif"))
        if not raw_files:
            print(f"Warning: No TIFF files found in {input_dir}")
            return False

        existing = glob.glob(os.path.join(output_dir, "*.tif"))
        if not force_reclip and len(existing) >= len(raw_files) * 0.9:
            print(f"{os.path.basename(input_dir)}: already clipped, skipping")
            return True

        print(f"Clipping {len(raw_files)} files from {os.path.basename(input_dir)}...")

        success_count = 0
        for f in tqdm(raw_files, desc="Clipping"):
            out_file = os.path.join(output_dir, os.path.basename(f))
            if os.path.exists(out_file) and not force_reclip:
                success_count += 1
                continue

            try:
                gdal.Warp(out_file, f, options=gdal.WarpOptions(
                    format='GTiff',
                    outputBounds=[self.lon_min, self.lat_min, self.lon_max, self.lat_max],
                    xRes=self.resolution,
                    yRes=self.resolution,
                    resampleAlg='bilinear',
                    dstSRS=self.projection,
                    outputType=gdal.GDT_Float32,
                    creationOptions=['COMPRESS=LZW', 'TILED=YES']
                ))
                success_count += 1
            except Exception as e:
                print(f"  Failed: {os.path.basename(f)} - {e}")

        print(f"Clipping complete for {os.path.basename(input_dir)}")
        return success_count > 0

    def prepare_all_sources(self, source_paths, clipped_paths):
        """
        Prepare all data sources by clipping each to the reference grid.

        Parameters
        ----------
        source_paths : dict
            Dictionary mapping source names to input directories.
        clipped_paths : dict
            Dictionary mapping source names to output directories.

        Returns
        -------
        bool
            True if all sources were prepared successfully.
        """
        all_success = True
        for name, in_path in source_paths.items():
            out_path = clipped_paths.get(name)
            if out_path is None:
                continue
            if not self.clip_to_grid(in_path, out_path):
                print(f"Warning: {name} preparation failed")
                all_success = False
        return all_success
