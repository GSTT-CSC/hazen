Getting started
===============
.. attention:: These guidelines are written for developers to install *hazen* on MacOS or Linux and are a work in progress!

If you would like to use with *hazen* via the CLI or contribute to its development, this is the installation guide for you.

If, however, you are interested in using or contributing to the *hazen web app*, please visit that project's GitHub repository `here <https://github.com/GSTT-CSC/hazen-web-app>`_ for its installation guide.

Prerequisites
-------------

* Python v3.9
* Git
* Docker

Install
-------

Clone the repository
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/GSTT-CSC/hazen.git

Docker
^^^^^^^^^^^^^^
We recommend using the Docker version of *hazen* as it is easy to get up-and-running and is linked to the most stable release. Refer to the `Docker installation instructions <https://docs.docker.com/engine/install>`_ to install Docker on your host computer.

For ease of use, it is recommended to copy the ``hazen-app`` script to a location accessible on the path such as ``/usr/local/bin``. This will allow you to run hazen from any location on your computer. Then, to use Docker hazen, simply run the ``hazen-app`` script appended with the function you want to use (e.g.: ``snr``).

In Terminal:

.. code-block:: bash

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

Linux & MacOS
^^^^^^^^^^^^^
For developers, hazen can be installed using ``pip``. We highly recommend using a virtual environment.

.. code-block:: bash

   # Install OpenSSL
   brew update
   brew upgrade
   brew install openssl
   export LDFLAGS="-L`brew --prefix openssl`/lib"
   export CPPFLAGS="-I`brew --prefix openssl`/include"

   # For M1 Apple Macs, also install OpenBLAS and LAPACK
   brew install openblas
   export LDFLAGS="-L/usr/local/opt/openblas/lib"
   export CPPFLAGS="-I/usr/local/opt/openblas/include"
   export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig"
   brew install lapack
   export LDFLAGS="-L/usr/local/opt/lapack/lib"
   export CPPFLAGS="-I/usr/local/opt/lapack/include"
   export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig"

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

Usage
-----

Command Line
^^^^^^^^^^^^
The CLI version of hazen is designed to be pointed at single folders containing DICOM file(s). Example datasets are provided in the `tests/data/` directory. If you are not using the Docker version of hazen, replace `hazen-app` with `hazen` in the following commands.

To perform an SNR measurement on the provided example Philips DICOMs:

.. code-block:: bash

   hazen-app snr tests/data/snr/Philips

To perform a spatial resolution measurement on example data provided by the East Kent Trust:

.. code-block:: bash

   hazen-app spatial_resolution tests/data/resolution/philips

To see the full list of available tools, enter:

.. code-block:: bash

   hazen-app -h

The ``--report`` option provides additional information for some of the functions. For example, the user can gain additional insight into the performance of the snr function by entering:

.. code-block:: bash

   hazen-app snr tests/data/snr/Philips --report

Web interface
^^^^^^^^^^^^^
Please refer to the *hazen web app* GitHub repository `here <https://github.com/GSTT-CSC/hazen-web-app>`_ for more information.

Contributing
------------

* The Release Manager should create a release branch for the future planned release e.g. release-X.X.X
* The RMs shall organise backlog refinement sessions to ensure issues are allocated to the appropriate release
* The RM should ensure their release branch is kept up-to-date with master
* PRs should be merged into the appropriate release branch for the issue(s) it is addressing

If you would like to contribute to the development of *hazen*, please take a look at the `Contributing`_ page.

Users
-----
Please `raise an Issue <https://github.com/GSTT-CSC/hazen/issues>`_ if you have any problems installing or running *hazen*.

We have used *hazen* with MRI data from a handful of different MRI scanners, including multiple different vendors. If your MRI data doesn't work with *hazen*, or the results are unexpected, please submit an Issue and we will investigate.

Releasing
---------

The Release Manager should ensure:
* All outstanding issues for the current release have been closed, or, transferred to future release.
* All tests are passing on Github Actions.
* All documentation has been updated with correct version numbers:

  * Version number in ``docs/conf.py``
  * Version number in ``hazenlib/__init__.py``
  * Version number in ``CITATION.cff``

* The ``release`` branch has been merged into ``main`` branch
* A new release has been created with a new version tag (tag = version number)
* RMs of other branches should update their release from the latest release as soon as possible and deal with any merge conflicts.
* RMs: Tom Roberts, Lucrezia Cester