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

Hazen How Tos
------------------------

The following list contains guidance for beginner programmers starting up using Hazen.

Get hazen running from terminal:

- Source hazen-venv/bin/activate
- Cd hazen
- Then should be able to run commands

Run Hazen Tests

- In terminal with hazen activated, type pytest tests

Commit your changes when you are done on a branch:

- Open Sourcetree
- On your task branch
- Select the files you want to send over
- Hit ‘commit’
- Write a helpful commit message in the format shown in the contributing guidelines
- Make sure the ‘push changes’ tick box is ticked
- Commit

To download any updates:

- Open terminal
- git checkout master
- git pull
- python setup.py install

How to stash changes made on your task branch

- On your task branch
- Go to source tree
- Stash (top ribbon bar)
- Write yourself a helpful message
- Click ‘stash’
- Then switch to master branch

To get back from master to stash

- Go to your task branch
- Go across to stashes (bottom of left hand column)
- Click on your changes
- Right click ‘apply stash’
- Click back on your branch and see your old changes

How to run your current branch from the terminal

- source hazen-venv/bin/activate
- Pip uninstall hazen
- Cd hazen
- Python setup.py develop

What to do when your task branches are shown as ‘X behind’
- Switch to branch
- ‘Pull’

Edit a commit message that was sent with the wrong wording (from https://linuxize.com/post/change-git-commit-message/ )

- If it is the most recent commit, go to terminal and type in
- git commit --amend -m "New commit message."
- git push --force branch-name

