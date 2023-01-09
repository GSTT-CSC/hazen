<!-- PROJECT HEADING -->
<br />
<p align="center">
<img src="https://raw.githubusercontent.com/GSTT-CSC/hazen/readme-dev/docs/assets/ibn-al-haytham.jpeg" alt="Ibn Al-Haytham">
</p>   
<h1 align="center">hazen</h1>
<p align="center">
Quality assurance framework for Magnetic Resonance Imaging
<br />
<a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
<br />
<br />
<a href="https://github.com/GSTT-CSC/hazen">View repo</a>
·
<a href="https://github.com/GSTT-CSC/hazen/issues">Report Bug</a>
·
<a href="https://github.com/GSTT-CSC/hazen/issues">Request Feature</a>
</p>
<p align="center">
<img src="https://github.com/GSTT-CSC/hazen/actions/workflows/tests_release.yml/badge.svg?branch=main">
<img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/laurencejackson/ba102d5f3e592fcd50451c2eff8a803d/raw/hazen_pytest-coverage-comment.json">
</p>
<p align="center">Please <b>STAR</b> this repo to receive updates about new versions of hazen!</p>

---

## Overview

hazen is a software framework for performing automated analysis of magnetic resonance imaging (MRI) Quality Assurance data.

It provides automatic quantitative analysis for the following measurements of MRI phantom data:
- Signal-to-noise ratio (SNR)
- Spatial resolution
- Slice position and width
- Uniformity
- Ghosting
- MR Relaxometry

Some example outputs from hazen:

| hazen snr                        | hazen ghosting                |
|----------------------------------|-------------------------------|
| ![](docs/assets/snr-example.png) | ![](docs/assets/ghosting.png) |

---

## Installation

### pip

Hazen can be installed using pip on python 3.9 or higher, it is recommended to use a virtual environment

```bash
python3 -m venv hazen-venv
source hazen-venv/bin/activate
pip install hazen
```

#### Docker

The Docker version of hazen as it is easy to get up-and-running and is linked to the most stable release. Refer to the [Docker installation instructions](https://docs.docker.com/engine/install) to install Docker on your host computer.

For ease of use, it is recommended to copy the `hazen-app` script to a location accessible on the path such as `/usr/local/bin`. This will allow you to run hazen from any location on your computer. Then, to use Docker hazen, simply run the `hazen-app` script appended with the function you want to use (e.g.: `snr`). 

In Terminal:

```bash 
cd hazen
cp ./hazen-app /usr/local/bin

# run hazen with CLI arguments
hazen-app snr tests/data/snr/Siemens/

latest: Pulling from gsttmriphysics/hazen
Digest: sha256:18603e40b45f3af4bf45f07559a08a7833af92a6efe21cb7306f758e8eeab24a
Status: Image is up to date for gsttmriphysics/hazen:latest
docker.io/gsttmriphysics/hazen:latest
{   'snr_smoothing_measured_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 191.16,
    'snr_smoothing_measured_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_3_1': 195.58,
    'snr_smoothing_normalised_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 1866.09,
    'snr_smoothing_normalised_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_3_1': 1909.2,
    'snr_subtraction_measured_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 220.73,
    'snr_subtraction_normalised_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 2154.69}
```


#### Linux & MacOS

For developers, hazen can be installed from source using `pip`. We highly recommend using a virtual environment.

```bash
# Install OpenSSL
# Go to local hazen repo directory
cd hazen

# Create and activate a virtual environment
python3 -m venv ./hazen-venv
source hazen-venv/bin/activate

# Install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install hazen
pip install .

# Run tests to ensure everything is working
pytest tests/
```

---

## Usage

### Command Line

The CLI version of hazen is designed to be pointed at single folders containing DICOM file(s). Example datasets are provided in the `tests/data/` directory. If you are not using the Docker version of hazen, replace `hazen-app` with `hazen` in the following commands.

To perform an SNR measurement on the provided example Philips DICOMs:

`hazen-app snr tests/data/snr/Philips`

To perform a spatial resolution measurement on example data provided by the East Kent Trust:

`hazen-app spatial_resolution tests/data/resolution/philips`

To see the full list of available tools, enter:

`hazen-app -h`

The `--report` option provides additional information for some of the functions. For example, the user can gain additional insight into the performance of the snr function by entering:

`hazen-app snr tests/data/snr/Philips --report`

### Web Interface

WIP: we are developing a web interface for hazen.

---

## Contributing
- The Release Manager should create a release branch for the future planned release e.g. release-X.X.X
- The RMs shall organise backlog refinement sessions to ensure issues are allocated to the appropriate release
- The RM should ensure their release branch is kept up-to-date with master
- PRs should be merged into the appropriate release branch for the issue(s) it is addressing

If you want to contribute to the development of hazen, please take a look at: `CONTRIBUTING.md`.

---

## Users

Please [raise an Issue](https://github.com/GSTT-CSC/hazen/issues) if you have any problems installing or running hazen.

We have used hazen with MRI data from a handful of different MRI scanners, including multiple different vendors. If your MRI data doesn't work with hazen, or the results are unexpected, please submit an Issue and we will investigate. 

---

## Releasing

The Release Manager should ensure:
- All outstanding issues for the current release have been closed, or, transferred to future release.
- All tests are passing on GitHub Actions.
- All documentation has been updated with correct version numbers:
   - Update version number `hazenlib/_version.py`, i.e. imported into `docs/conf.py`, `hazenlib/__init__.py` and `setup.cfg`
   - Update version number in `CITATION.cff`
- The `release` branch has been merged into `main` branch
- A new release has been created with a new version tag (tag = version number)

- RMs of other branches should update their release from the latest release as soon as possible and deal with any merge conflicts.

![image](https://user-images.githubusercontent.com/19840489/143266366-06e33949-12c7-44b4-9ed7-c0a795b5d492.png)

- RMs: Tom Roberts, Lucrezia Cester

