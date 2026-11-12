# eivd-fusion
Code for "A 1980-2020 Daily Meteorological Dataset over China for Near-Surface Variables". EIVD-based fusion framework merging ERA5-Land, MERRA-2, and JRA-3Q with CMA station observations.
# EIVD Meteorological Data Fusion Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GDAL](https://img.shields.io/badge/GDAL-3.0%2B-orange.svg)](https://gdal.org/)

Python implementation of the Extended Double Instrumental Variable (EIVD) method for merging reanalysis products with station observations into high-quality meteorological forcing datasets.

This framework is developed for generating the dataset described in:

> Yang, Y., Zhou, H., Tian, X., Ding, L., Hu, Y., & Dong, J. (2026). A 1980-2020 Daily Meteorological Dataset over China for Near-Surface Variables. *Scientific Data*.

## Features

- **EIVD-based error estimation** – Estimates error variances and covariances using lag-1 auto-covariances as instrumental variables
- **BLUE optimal weighting** – Computes optimal merging weights within the Best Linear Unbiased Estimation framework
- **Monthly bias correction** – Corrects systematic biases using station-interpolated reference fields
- **Two-step wind merging** – Preserves vector wind information using station U/V ratios
- **Variable-specific handling** – Supports temperature unit conversion, MERRA-2 RH from dew point, and pressure unit conversion
- **Modular design** – Clean separation of preprocessing, core algorithm, and variable-specific logic

## Variables Supported

| Variable | Unit |
|----------|------|
| Temperature (2-meter) | K |
| Relative Humidity | % |
| Surface Pressure | hPa |
| Wind Speed (10-meter, with U/V components) | m/s |

## Repository Structure
eivd-fusion/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── preprocessing/
│   ├── __init__.py
│   ├── resample.py
│   └── bias_correction.py
│   └── station_interpolation.py
│   └── station_sampling.py
├── eivd_core/
│   ├── __init__.py
│   ├── eivd_base.py
│   ├── eivd_temperature.py
│   ├── eivd_humidity.py
│   ├── eivd_pressure.py
│   └── eivd_wind.py
└── utils/
    ├── __init__.py
    ├── io_utils.py
    ├── grid_utils.py
    └── validation.py


## Data Requirements

Each fusion module requires four input data sources:

| Data Source | Description |
|-------------|-------------|
| ERA5-Land | Reanalysis product from ECMWF |
| MERRA-2 | Reanalysis product from NASA |
| JRA-3Q | Reanalysis product from JMA |
| Calibration | Station-interpolated reference field from CMA |

A reference GeoTIFF is required to define the target spatial grid. All inputs are resampled to match this grid.

## Methodology

### Extended Double Instrumental Variable (EIVD)

**Error Model [Eq.(4)]:**
X_i = α_i · p + β_i + ε_i

text

**Error Covariance Matrix [Eq.(5)]:**
Σ = [[σ_x², 0, 0 ],
[ 0, σ_w², σ_wz],
[ 0, σ_wz, σ_z²]]


**BLUE Optimal Weights [Eq.(7)]:**
w* = (Σ⁻¹ · 1) / (1ᵀ · Σ⁻¹ · 1)



**Merged Product [Eq.(8)]:**
X_merged = w_x* · X_corr_x + w_w* · X_corr_w + w_z* · X_corr_z



### Two-Step Wind Merging

**Step 1 – Merge wind speed magnitude [Eq.(8)]**

**Step 2 – Decompose U/V components using station observations [Eq.(3)]:**
U_merged = WS_merged × (U_sta / WS_sta)
V_merged = WS_merged × (V_sta / WS_sta)

text

## Output Structure
output_directory/
├── clipped_data/ # Clipped inputs
├── daily_results/ # Fused daily products
│ ├── merged_temperature_YYYYMMDD.tif
│ ├── merged_humidity_YYYYMMDD.tif
│ ├── merged_pressure_YYYYMMDD.tif
│ └── merged_windspeed_YYYYMMDD.tif
├── eivd_global_weights.pkl # EIVD weights
├── fusion_log.json # Processing log
├── validation_results.json # Validation metrics
└── diagnostics_report.txt # Diagnostic report



## Dependencies

- Python >= 3.8
- NumPy >= 1.20
- GDAL >= 3.0 (with Python bindings)
- tqdm >= 4.60

## Reference

If you use this code, please cite:

```bibtex
@article{yang2026eivd,
  author = {Yang, Y. and Zhou, H. and Tian, X. and Ding, L. and Hu, Y. and Dong, J.},
  title = {A 1980-2020 Daily Meteorological Dataset over China for Near-Surface Variables},
  journal = {Scientific Data},
  year = {2026}
}

@article{dong2020eivd,
  author = {Dong, J. and Wei, L. and Chen, X. and Duan, Z. and Lu, Y.},
  title = {An instrument variable based algorithm for estimating cross-correlated hydrological remote sensing errors},
  journal = {Journal of Hydrology},
  volume = {581},
  pages = {124413},
  year = {2020},
  doi = {10.1016/j.jhydrol.2019.124413}
}
License
MIT

Contact
Yuze Yang – yangyuze@stu.ynu.edu.cn
School of Earth System Science, Tianjin University
