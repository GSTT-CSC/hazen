.. _Contributing:

Get involved
=================

If you would like to contribute to the development of *hazen* and get involved with the project, we suggest you:

* Start by creating a new issue for the feature and:

  * decide which release this is for
  * gather any user research
  * define acceptance test criteria
* Create a Git feature-branch for this ticket
* Create an empty file detailing the design in the file docstring
* Create an empty file detailing unit, integration and system tests, where appropriate
* Create pull request and get the design approved
* Code according to approved design and acceptance criteria
* Keep an eye on the feedback from Bitbucket Pipelines and Sonarcube
* Check a running instance of your app by locally running the latest Docker:

  .. code-block:: docker

     image:docker run --rm -p5000:5000 -it

* Once you’re satisfied with the new feature, the pull request can be formally reviewed for merging

Changing DB models
------------------

Remember to to generate migration script by running:

.. code-block:: bash

 __flask db migrate -m "very short commit message"__

Sonarcube
---------

To run local, install sonar-scanner. Edit properties file in conf so that url is pointing to cloud instance.

Branching
---------

* Git-flow.

Merging
-------

* Pull request.
* Haris is final reviewer.

Releasing
---------

* Dependencies are managed using ``uv`` and specified in ``pyproject.toml``
* The ``uv.lock`` file is automatically maintained by ``uv`` and should be committed to the repository
* Make sure all tests are passing
* Update version in ``hazenlib/__init__.py``, remove ``dev``.
* Update docs (we use ``sphinx-apidoc`` for autodocumentation of modules)


