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

   # change to your hazen directory if you're not already in it
   $ cd hazen

   # run help command to see what commands are available to you
   $ hazen --help

   # as an example, this is how to perform SNR measurements on data
   $ hazen snr /path/to/snr/dicom/directory

Running Tests
----------------------

Hazen comes with built-in tests that check installation has worked correctly. You can run these tests by typing the following into your command window:

.. code-block:: bash

   $ pytest tests


Updating Hazen
----------------------
You can update Hazen from your terminal using the following:

.. code-block:: bash

   $ git checkout master
   $ git pull
   $ python setup.py install




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

See doc:`../guides/guides` on how to begin contributing your code

Developer Hazen How Tos
------------------------

The following list contains guidance for beginner programmers starting up using Hazen.

Commit your changes when you are done on a branch:

- Open Sourcetree
- On your task branch
- Select the files you want to send over
- Hit ‘commit’
- Write a helpful commit message in the format shown in the contributing guidelines
- Make sure the ‘push changes’ tick box is ticked
- Commit


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

.. code-block:: bash

   $ source hazen-venv/bin/activate
   $ pip uninstall hazen
   $ cd hazen
   $ python setup.py develop

What to do when your task branches are shown as ‘X behind’
- Switch to branch
- ‘Pull’

Edit a commit message that was sent with the wrong wording (from https://linuxize.com/post/change-git-commit-message/ )

- If it is the most recent commit, go to terminal and type in

.. code-block:: bash

   $ git commit --amend -m "New commit message."
   $ git push --force branch-name
