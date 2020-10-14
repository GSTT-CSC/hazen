***************
Getting started
***************

Users
*****

Pre-installation requirements
-----------------------------

You will need the following packages before installing Hazen:

- Python3.6
- Celery
- RabbitMQ
- Postgres
- Postico (recommended)


Python
######
Hazen was created with Python3.6 and will not work with newer versions. Check which version you're running:

.. code-block:: bash

    $ python -V

If your version is different from 3.6, use Pyenv to switch to Python3.6:

.. code-block:: bash

    # Install pyenv
    $ brew install pyenv

    # Install Python3.6
    $ pyenv install python3.6

    # Set local Python version to 3.6
    $ pyenv local 3.6

    # Check it worked
    $ python -V

Celery, RabbitMq, Postgres
##########################

Celery will work as a task manager and needs RabbitMQ to act as the message broker. Postgres will provide the database. It is recommended you use Postico to manage your databases.

To install and run Celery:

.. code-block:: bash

    $ pip install celery
    $ celery -A project worker --loglevel=info

To install and run RabbitMQ:

.. code-block:: bash

    $ brew install rabbitmq
    $ rabbitmq-server

To install Postgres:

.. code-block:: bash

    $ brew install postgresql
    $ psql postgres

To install Postico, visit https://eggerapps.at/postico/ Once installed, create two databases:

- hazen
- hazen_test

In hazen_test database, add the following SQL query:

.. code-block:: bash

    CREATE USER test_user WITH PASSWORD 'test_user_password'

Click on 'Execute statement'. It should say CREATE ROLE in the under the query box. Close Postico.

Hazen installation
------------------
- Make sure you have an account on https://bitbucket.org
- From your Terminal application:

.. code-block:: bash

    $ git clone https://<BITBUCKET_USERNAME>@bitbucket.org/gsttmri/hazen.git

    # Create and activate a virtual environment
    $ python3 -m venv ./hazen-venv
    $ source hazen-venv/bin/activate
    $ cd hazen

    # Install requirements
    $ pip install --upgrade pip
    $ pip install -r requirements.txt

    # Install hazen
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

To run the web app from terminal:

.. code-block:: bash

    $python hazen.py

Requirements/Recommended
------------------------
- MacOS/Unix
- Python3.6
- Git
- Pycharm (recommended)
- Sourcetree (recommended)
- Docker (recommended)

See doc:``../guides/guides`` on how to begin contributing your code