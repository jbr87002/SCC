# SCC plotting

(c) Jonathan Goodman, Richard Lewis, Benji Rowlands  
2024  
jmg11@cam.ac.uk  
University of Cambridge  

## Overview

This repository contains scripts to reproduce the SCC plots from the paper
"Towards automatically verifying chemical structures: the powerful combination
of <sup>1</sup>H NMR and IR spectroscopy". 

The main plotting script is contained in `scc.py`. An example input file is
contained in `config.yaml`.

## Requirements
- Operating system: this software has been tested on macOS Sonoma 14.5. It
  should work on other operating systems, but has not been tested on them.
- Python version: this software was developed using Python 3.12.

## Installation

To run this software, ensure you have Python 3.12 installed along with the
following packages:
- `matplotlib` (version 3.9.0)
- `pandas` (version 2.2.2)
- `numpy` (version 2.0.1)
- `scipy` (version 1.14)
- `scikit-learn` (version 1.5.1)
- `scienceplots` (version 2.1.1)
- `pyyaml` (version 6.0.2)
- `PyQt5` (version 5.15.11)

Installation time will depend on how long it takes to download the necessary
packages, but should be less than a minute.

To configure the environment correctly, follow the following steps:
1. **Install Python**: Ensure you have Python 3.12 installed. You can
   download it from the [official Python
   website](https://www.python.org/downloads/release/python-3124/).

2. **Create a virtual environment**:
    ```
    python3.12 -m venv .venv
    source .venv/bin/activate 
    ```

3. **Activate the virtual environment and install the required package**:
    ```
    source .venv/bin/activate
    pip install matplotlib==3.9.0 pandas==2.2.2 numpy==2.0.1 scipy==1.14 scikit-learn==1.5.1 scienceplots==2.1.1 pyyaml==6.0.2 pyqt5==5.15.11
    ```

## Usage

### SCC plots

1. **Prepare your input file**: The input file contains the settings for the
   plot, and should be named `config.yaml`. An example input file which
   reproduces the plots in the paper is provided.

2. **Run the script**: Execute the script from the command line. With no
   modification to the input file provided, a plot will be generated using data
   from the `data/` directory.
```bash
python scc.py config.yaml --outdir ./plots
```

Execution time should be a few seconds.

## Output

`scc.py` will save the plots to the `plots/` directory.

## Data
`data` contains 9 files:
- `DP4_H.csv` contains DP4* scores for the molecules in the test set
- `ACD.csv` contains ACD scores for the molecules in the test set
- `ACD_auto.csv` contains ACD scores for the molecules in the test set, obtained
  using ACD's automatic peak-picking tool
- `IR_high_fwhm_8.csv` contains IR.Cai scores for the molecules in the test set,
  obtained using high-level theory and a fwhm of 8cm<sup>-1</sup>
- `IR_high_fwhm_10.csv` contains IR.Cai scores for the molecules in the test
  set, obtained using high-level theory and a fwhm of 10cm<sup>-1</sup>
- `IR_high_fwhm_12.csv` contains IR.Cai scores for the molecules in the test
  set, obtained using high-level theory and a fwhm of 12cm<sup>-1</sup>
- `IR_low_fwhm_8.csv` contains IR.Cai scores for the molecules in the test set,
  obtained using low-level theory and a fwhm of 8cm<sup>-1</sup>
- `IR_low_fwhm_10.csv` contains IR.Cai scores for the molecules in the test set,
  obtained using low-level theory and a fwhm of 10cm<sup>-1</sup>
- `IR_low_fwhm_12.csv` contains IR.Cai scores for the molecules in the test set,
  obtained using low-level theory and a fwhm of 12cm<sup>-1</sup>

The data are in csv format. The 'Molecule' column identifies the molecule in
question from the test set. The 'Comparison' column identifies which isomers are
being compared. The '0', '1', '2', and '3' columns contain the scores for the
relevant isomer for each molecule. As throughout this work, the score in the '0'
column is the score for the correct isomer; the other columns contain scores for
incorrect isomers.

## Contact

For any questions or issues, please contact Jonathan Goodman at jmg11@cam.ac.uk.