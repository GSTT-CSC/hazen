***************
Getting started
***************

Users
*****

Installation
------------

- Make sure you have an account on https://bitbucket.org
- From your Terminal application:
.. code-block:: bash

   $ git clone https://<BITBUCKET_USERNAME>@bitbucket.org/gsttmri/hazen.git

   # Create and activate a virtual environment
   $ python3 -m venv ./hazen-venv
   $ source hazen-venv/bin/activate

   # Install requirements
   $ pip install --upgrade pip
   $ pip install -r requirements.txt

    # Install hazen
   $ cd hazen
   $ python setup.py install

   # Run tests to make sure everything is working
   $ pytest tests/

Command-Line Interface
----------------------

Hazen includes a command-line interface, that allows you to run hazen tasks without needing the website:

.. code-block:: bash

   # activate virtual environment
   $ source ../hazen-venv/bin/activate

   # run help command to see what commands are available to you
   $ hazen --help

   # as an example, this is how to perform SNR measurements on data
   $ hazen snr /path/to/snr/dicom/directory


Developers
**********

Steps to take to get started:

Requirements/Recommended
------------------------
- MacOS/Unix
- Python3.7
- Git
- Pycharm (recommended)
- Sourcetree (recommended)
- Docker (recommended)

See :doc:`../guides/guides` on how to begin contributing your code