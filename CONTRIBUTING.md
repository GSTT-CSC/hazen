## Table of Contents
- [1) Introduction](#1-introduction)
- [2) How to make and test code changes](#2-how-to-make-and-test-code-changes)
- [3) Developer Process for Contributing](#3-developer-process-for-contributing)
- [4) Continuous Integration (CI)](#4-continuous-integration-ci)
- [5) Release Process](#5-release-process)
- [6) Update Documentation](#6-update-documentation)


## 1) Introduction

Welcome to hazen! This documentation is intended for individuals and developers interested in contributing to hazen. We 
anticipate contributions in the following main areas:

1. **Enhancements**: General hazen functionality, performance and user experience
2. **Bugfixes**: Fixing issues with existing hazen code
3. **MRI**: MR image processing methods
4. **DICOM**: DICOM file and metadata manipulation
5. **Documentation**: Improvements to user guidance

## 2) How to make and test code changes

Clone and install this repo following the guidance below. This requires git and Python 3.11+ installed and accessible 
within your PATH. We highly recommend using [uv](https://docs.astral.sh/uv/) for development and testing.

Where possible, make small granular commits (rather than singular large commits!) with descriptive messages. Please 
separate feature enhancements and bugfixes into individual branches for easier review.

### Using uv (recommended)

```bash
# Install uv if you haven't already
# On macOS and Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Clone hazen repo (GSTT-CSC)
# - this will create a folder named 'hazen' in the current working directory
git clone https://github.com/GSTT-CSC/hazen.git

# Or, for hazen-wales use the following instead (and not the previous command):
git clone git@github.com:sbu-physics-mri/hazen-wales.git

# Go to local copy of the repo
# - use 'cd hazen' for the GSTT-CSC repo, or 'cd hazen-wales' for the hazen-wales fork
cd hazen

# Install hazen and its dependencies (including dev dependencies)
uv sync --group dev

# Run tests to ensure everything is working on your local machine, prior to development
# They take a while to run all of the tests so we recommend making a cup of tea at this point.
uv run pytest tests/

# After making a code change, run tests on the relevant code you have edited, e.g.:
uv run hazen snr tests/data/snr/GE
# You can also run specific Tasks or scripts without installing the module by directly executing the local file, e.g.:
uv run python hazenlib/__init__.py snr tests/data/snr/GE
```


## 3) Developer Process for Contributing

Follow these steps to make a contribution to hazen:

1. Check the current [Issues](https://github.com/GSTT-CSC/hazen/issues) to see if an Issue already exists for your 
contribution - alternatively check [Hazen Wales' Issues](https://github.com/sbu-physics-mri/hazen-wales/issues) to see if your issue features there.
2. If there is no existing Issue, create a new Issue for your contribution:
   - Select the `Bug report` or `Feature request` template
   - Fill in the Issue according to the template
   - Add relevant Labels to the issue: `Enhancement`, `Bug`, `MRI`, `DICOM`, etc
3. **Contact us to be given 'write' access to the repo** - this will allow you to branch off (note: we cannot merge from forks). Create a new branch from `main`
   - Name the branch with the issue number and a short description, e.g.: `123-snr-bugfix`
4. Make your code changes (see guidance above)
5. Perform unit tests on your machine: `uv run pytest tests/`
6. Create a [Pull Request](https://github.com/GSTT-CSC/hazen/pulls) (PR)
   - Describe your changes
   - Describe why you coded your change this way
   - Describe any shortcomings or work still to do
   - Cross-reference any relevant Issues
7. One of the hazen team members will review your PR and then:
   - Ask questions
   - Request any changes
   - Merge into `main` – thank you and congratulations on contributing to hazen!


## 4) Continuous Integration (CI)

hazen uses a three-tiered CI strategy to balance fast feedback with comprehensive testing. All CI workflows
use Makefile targets as the single source of truth for how to run tasks, ensuring consistency between local
development and CI.

### CI Tiers

| Tier | Trigger | Duration | Purpose | Make Target |
|------|---------|----------|---------|-------------|
| **Per-Commit** | Push to feature branches | < 1 min | Fast feedback | `make ci-commit` |
| **Pre-Merge** | Pull request to `main` | < 15 min | Comprehensive validation | `make ci-pr` |
| **Release** | Push to `main` or `release/*` | < 30 min | Exhaustive verification | `make ci-release` |

### What Each Tier Runs

- **Per-Commit (`make ci-commit`)**: Lint, format check, and fast unit tests (excludes slow tests)
- **Pre-Merge (`make ci-pr`)**: Lint, format check, type checking, tests with coverage, and CLI smoke tests
- **Release (`make ci-release`)**: Full check suite, comprehensive tests with coverage, and all CLI tests

### Reproducing CI Locally

You can reproduce the exact CI checks locally using the same Makefile targets:

```bash
# Run what CI runs on each commit (fast)
make ci-commit

# Run what CI runs on pull requests (comprehensive)
make ci-pr

# Run what CI runs on releases (exhaustive)
make ci-release
```

### Available Makefile Targets

Run `make help` to see all available targets. Key testing commands include:

- `make test-fast`: Run tests quickly without coverage (excludes slow tests)
- `make test`: Run tests with coverage
- `make test-ci`: Run tests with CI-compatible output (JUnit XML + coverage)
- `make test-cli-smoke`: Run essential CLI smoke tests
- `make lint`: Run ruff linter
- `make format-check`: Check code formatting
- `make type-check`: Run type checkers (mypy and ty)


## 5) Release Process

The Release Process involves approving and then merging all PRs identified for the new release of hazen. 
Follow these steps for a new Release:

For each PR:
1. Ensure you understand the changes being introduced by the PR
   - Request further information or changes from the Developer where necessary 
2. When the PR is ready to merge, ensure all automated GitHub tests and checks are passing
   - Investigate any which are failing
3. Merge the PR into the `main` branch
   - Provide a brief description of changes in the PR textbox as this forms the Release Notes 

For a new release: <br>

4. Close all related Issues resolved by the merged PRs
5. **Important**: Update version numbers across the repo:
   - Update version number in `pyproject.toml`
     - This is automatically propagated into across the repository.
   - Update version number and date released in `CITATION.cff`
   - Updated contributors in `docs/source/contributors.rst`
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
   - Below is an example email from a previous release. 
   - The "What's Changed" section is copy-pasted from the 
   [corresponding Release page](https://github.com/GSTT-CSC/hazen/releases/tag/0.5.2).

> **Subject:** hazen 0.5.2 released
   
> Dear hazen users,
> 
> We have released hazen version 0.5.2: 
>
> https://github.com/GSTT-CSC/hazen/releases/tag/0.5.2
>
> We have improved the ghosting function to be more robust, the rod centroid estimates within the 
> slice_width function are now more accurate which provides better linearity measurements (thanks Sian!) and Paul’s 
> snr_map function is now part of hazen too, which provides an SNR map across the phantom – useful for seeing 
> signal dropout/fluctuations. Also some under the hood improvements.
>
> Full changelog below. Let us know any issues.
> 
> Thanks,
> 
> [_name of Release Master_]
>
> ---
> ## 0.5.2
> ### What's Changed
> - Updated ghosting function – fixes sampling point issue by @Lucrezia-Cester in #185
> - Added new hazen snr_map function – generates SNR parametric map by @pcw24601 in #113
> - Corrects hazen slice_width geometric distortion measurements by @superresolusian @tomaroberts in #175
> - Updated LSF function by @Lucrezia-Cester in #191
> - Added hazenlib init tests coverage by @Lucrezia-Cester in #144
> - Creates citation button on Github page, updates contributors.txt by @tomaroberts in #194
> - Updated cli-test.yml by @laurencejackson in #195
> - Release/0.5.2 by @tomaroberts in #196

## 6) Update Documentation

Create rst files describing the structure of the hazen Python Package

```
# in an active hazen virtual environment in the root of the project
# the command below specifies that sphinx should look for scripts in the hazenlib folder
# and output rst files into the docs/source folder
uv run sphinx-apidoc -o docs/source hazenlib

# next, from within the docs/ folder
cd docs/
# create/update the html files for the documentation
uv run make html  -f Makefile
# opening the docs/source/index.html in a web browser allows a preview of the generated docs
```
