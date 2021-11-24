<!-- PROJECT HEADING -->
<br />
<p align="center">
<a href="https://github.com/github_username/repo_name">
    <img src="https://raw.githubusercontent.com/GSTT-CSC/gstt-csc.github.io/main/assets/transparent-CSC-logo-cropped.png" alt="Logo" width="40%">
  </a>
<h1 align="center">Hazen</h1>
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
  <img src="https://github.com/GSTT-CSC/hazen/actions/workflows/tests_release.yml/badge.svg?branch=master">
  <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/laurencejackson/ba102d5f3e592fcd50451c2eff8a803d/raw/hazen_pytest-coverage-comment.json">
</p>

## Overview

Please 'star' this repository to receive release updates!




## Usage

---
### Docker
To use the docker version of hazen simply run the `hazen-app` script in a terminal. Docker must be installed on the host 
system for this method to work, see [docker installation instructions](https://docs.docker.com/engine/install).
For ease of use it is recommended to copy the hazen-app script to location accessible on the path such as `/usr/local/bin` 
so you can run it from any location.

e.g.
```bash
cp ./hazen-app /usr/local/bin

./hazen-app snr tests/data/snr/Siemens/

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

## Releasing
The Release Manager should ensure:
- All outstanding issues for that release have been closed or transferred to future release
- All tests are passing on Github Actions
- All documentation has been updated included version numbers
- Update version number in `hazenlib/__init__.py`
- Merge the release branch into master
- Create release on Github with new version tag (tag = version number)

- RMs of other branches should update their release from the new master release as soon as possible and deal with any merge conflicts.


![image](https://user-images.githubusercontent.com/19840489/143266366-06e33949-12c7-44b4-9ed7-c0a795b5d492.png)

- RMs: Tom Roberts, Lucrezia Cester


---

## Contributing
- The Release Manager should create a release branch for the future planned release e.g. release-X.X.X
- The RMs shall organise backlog refinement sessions to ensure issues are allocated to the appropriate release
- The RM should ensure their release branch is kept up-to-date with master
- PRs should be merged into the appropriate release branch for the issue(s) it is addressing

Read CONTRIBUTING.md

---

## Users

Nothing to see here. Maybe see hazen/docs.

---
