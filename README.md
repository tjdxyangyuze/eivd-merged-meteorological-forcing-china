# eivd-fusion
Code for "A 1980-2020 Daily Meteorological Dataset over China for Near-Surface Variables". EIVD-based fusion framework merging ERA5-Land, MERRA-2, and JRA-3Q with CMA station observations.
# EIVD Meteorological Data Fusion Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GDAL](https://img.shields.io/badge/GDAL-3.0%2B-orange.svg)](https://gdal.org/)

Python implementation of the Extended Double Instrumental Variable (EIVD) method for merging reanalysis products (ERA5-Land, MERRA-2, JRA-3Q) with station observations.

## Variables Supported

- Temperature (2-meter, K)
- Relative Humidity (%)
- Surface Pressure (hPa)
- Wind Speed (10-meter, m/s, with U/V components)

## Repository Structure
eivd-fusion/
├── requirements.txt
├── config.py # User configuration (paths and parameters)
├── main.py # Main entry point
├── preprocessing/ # Resampling and bias correction
│ ├── resample.py
│ └── bias_correction.py
├── eivd_core/ # EIVD fusion for each variable
│ ├── eivd_base.py
│ ├── eivd_temperature.py
│ ├── eivd_humidity.py
│ ├── eivd_pressure.py
│ └── eivd_wind.py
└── utils/ # I/O and validation utilities
├── io_utils.py
├── grid_utils.py
└── validation.py



## Quick Start

```bash
git clone https://github.com/tjdxyangyuzze/eivd-fusion.git
cd eivd-fusion
pip install -r requirements.txt
Edit config.py with your data paths, then run:

bash
python main.py
Dependencies
Python >= 3.8

NumPy >= 1.20

GDAL >= 3.0

tqdm >= 4.60

Reference
If you use this code, please cite:

bibtex
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
