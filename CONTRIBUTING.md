## How to contribute

### 1. Clone this repo for local development

Requires Python 3.9, pip and venv installed and on PATH. We highly recommend using a virtual environment for development and testing.

```bash
# Install OpenSSL - is it still needed? TODO
git clone https://github.com/GSTT-CSC/hazen...

# Go to local copy of hazen repo
cd hazen

# Create and activate a virtual environment
python3 -m venv ./hazen-venv
source hazen-venv/bin/activate

# Install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run tests to ensure everything is working
pytest tests/

# for development and testing, the hazen module may be installed after each change
pip install .
# and then running the relevant task, eg
hazen snr tests/data/snr/GE
# or a specific tasks/scripts can be run without installing the module
python hazenlib/__init__.py snr tests/data/snr/GE
```

---
### 2. Developer responsibilities
- Start by creating a new issue for the feature and:
    - gather any user research
    - define acceptance test criteria
    - decide which release this is for
- Create a Git feature-branch for this ticket
- Create an empty file detailing the design in the file docstring
- Create an empty file detailing unit, integration and system tests, as appropriate
- Create pull request and get the design approved
- Code according to approved design and acceptance criteria
- Keep an eye on the feedback from Bitbucket Pipelines and Sonarcube
- Check a running instance of your app by locally running the latest Docker image: `docker run --rm -p5000:5000 -it`
- Once you’re satisfied with the new feature, the pull request can be formally reviewed for merging


### 3. Release Manager responsibilities
- The Release Manager should create a release branch for the future planned release e.g. release-X.X.X
- The RMs shall organise backlog refinement sessions to ensure issues are allocated to the appropriate release
- The RM should ensure their release branch is kept up-to-date with master
- PRs should be merged into the appropriate release branch for the issue(s) it is addressing

## Contribution guidelines
### Sonarcube
To run local, install sonar-scanner. Edit properties file in conf so that url is pointing to cloud instance.

### Branching
- Git-flow.

### Merging

- Pull request. 
- Haris is final reviewer.

## Releasing

- Produce requirements.txt: __pipreqs --force --savepath ./requirements.txt --ignore bin,hazen-venv ./__
- Check what requirements have been edited as pipreqs is not perfect e.g. scikit_image instead of skimage
- Make sure all tests are passing
- Update version in hazenlib/\_\_init\_\_.py, remove 'dev'.
- Update docs (try sphinx-apidoc for autodocumentation of modules)
<br></br>

The Release Manager should ensure:
- All outstanding issues for the current release have been closed, or transferred to future release.
- All tests are passing on GitHub Actions.
- All documentation has been updated with correct version numbers:
   - Update version number `hazenlib/_version.py`, i.e. imported into `docs/conf.py`, `hazenlib/__init__.py` and `setup.cfg`
   - Update version number in `CITATION.cff`
- The `release` branch has been merged into `main` branch
- A new release has been created with a new version tag (tag = version number)

- RMs of other branches should update their release from the latest release as soon as possible and deal with any merge conflicts.

![image](https://user-images.githubusercontent.com/19840489/143266366-06e33949-12c7-44b4-9ed7-c0a795b5d492.png)

- RMs: Tom Roberts

