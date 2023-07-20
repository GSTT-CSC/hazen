## Table of Contents
- [Introduction](#introduction)
- [1) How to make and test code changes](#1-how-to-make-and-test-code-changes)
- [2) Developer Process for Contributing](#2-developer-process-for-contributing)
- [3) Release Process](#3-release-process)

## Introduction

Welcome to hazen! This documentation is intended for individuals and developers interested in contributing to hazen. We 
anticipate contributions in the following main areas:

1. **Enhancements**: General hazen functionality, performance and user experience
2. **Bugfixes**: Fixing issues with existing hazen code
3. **MRI**: MR image processing methods
4. **DICOM**: DICOM file and metadata manipulation
5. **Documentation**: Improvements to user guidance

## 1) How to make and test code changes

Clone and install this repo following the guidance below. This requires git, Python 3.9, pip and a venv installed and on
accessible within your PATH. We highly recommend using a virtual environment for development and testing.

Where possible, make small granular commits (rather than singular large commits!) with descriptive messages. Please 
separate feature enhancements and bugfixes into individual branches for easier review.

```bash
# Clone hazen repo
# - this will create a folder named 'hazen' in the current working directory
git clone https://github.com/GSTT-CSC/hazen.git

# Go to local copy of hazen repo
cd hazen

# Create and activate a virtual environment
# - using 'python' or 'python3' will depend on your local installation of Python
python3 -m venv ~/hazen-venv
source ~/hazen-venv/bin/activate

# Install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run tests to ensure everything is working on your local machine, prior to development
pytest tests/

# After making a code change, you will need to rebuild your local hazen install
pip install .
# optionally, use the -e flag to install the module in editable way, to avoid having to reinstall after each change
#    pip install -e .
# optionally, use the -q flag for quiet installation
#    pip install -e . -q

# Re-run the unit tests on the relevant code you have edited, e.g.:
hazen snr tests/data/snr/GE

# You can also run specific Tasks or scripts without installing the module by directly executing the local file, e.g.:
python hazenlib/__init__.py snr tests/data/snr/GE
```

## 2) Developer Process for Contributing

Follow these steps to make a contribution to hazen:

1. Check the current [Issues](https://github.com/GSTT-CSC/hazen/issues) to see if an Issue already exists for your 
contribution.
2. If there is no existing Issue, create a new Issue for your contribution:
   - Select the `Bug report` or `Feature request` template
   - Fill in the Issue according to the template
   - Add relevant Labels to the issue: `Enhancement`, `Bug`, `MRI`, `DICOM`, etc
3. Create a new branch from `main`
   - Name the branch with the issue number and a short description, e.g.: `123-snr-bugfix`
4. Make your code changes (see guidance above)
5. Perform unit tests on your machine: `pytest tests/`
6. Create a [Pull Request](https://github.com/GSTT-CSC/hazen/pulls)
   - Describe your changes
   - Describe why you coded your change this way
   - Describe any shortcomings or work still to do
   - Cross-reference any relevant Issues
7. One of the hazen team members will review your Pull Request and then:
   - Ask questions
   - Request any changes
   - Merge into `main` – thank you and congratulations on contributing to hazen!


## 3) Release Process

The Release Process involves approving and then merging all Pull Requests identified for the new release of hazen. 
Follow these steps for a new Release:

For each PR:
1. Ensure you understand the changes being introduced by the PR
   - Request further information or changes from the Developer where necessary 
2. When the PR is ready to merge, ensure all automated GitHub tests and checks are passing
   - Investigate any which are failing
3. Merge the PR into the `main` branch
   - Provide a brief description of changes in the PR textbox as this forms the Release Notes 

For a new release:
4. Close all related Issues resolved by the merged PRs
5. **Important**: Update version numbers across the repo:
   - Update version number in `hazenlib/_version.py`
     - (This is automatically propagated into `docs/source/conf.py`, `hazenlib/__init__.py` and `setup.cfg`)
   - Update version number in `CITATION.cff`
6. Create a [new Release](https://github.com/GSTT-CSC/hazen/releases)
   - Create a tag equal to the version number, e.g. 1.2.1
   - Select `main` as the Target branch from which to create the Release
   - Select "Generate release notes" to automatically pull in the Merge Commit messages
   - Amend the Release Notes:
     - Add any other changes not automatically generated
     - Edit Release Notes for typos and formatting consistency
7. Publish Release
   - This will trigger the [publishing GitHub Actions](https://github.com/GSTT-CSC/hazen/tree/main/.github/workflows) 
   which push the released version to DockerHub and [PyPI](https://pypi.org/project/hazen/).
8. Send a Release email to the hazen users!

