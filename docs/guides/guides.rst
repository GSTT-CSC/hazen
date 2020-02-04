******
How-to
******


Develop collaboratively with Git-Flow
#####################################

This project is developed loosely following the Git-Flow model expounded
`here <https://nvie.com/posts/a-successful-git-branching-model>`_.

Branching model
---------------
Principally, there are 5 types of branches in this project:

- Master:
    Only one instance of this branch exists in origin at any one time. This is the closest approximation to
    the live production code running on the server.

- Develop:
    Again, only one instance of this exists in origin at any one time. This branch represents the latest
    development code. New features should branch off this branch and should be merged back into this branch once they
    are completed.

- Feature:
    Every feature branch should branch from the latest develop code. Ideally, each branch has only one developer
    in order to minimise conflicts. Frequent committing (daily) is recommended in order to permit continuous peer-review

- Hotfix:
    Hotfixes are code changes that need to be implemented ASAP to restore some core functionality that was lost due to
    an unforeseen bug. It branches from the **Master** branch and is merged back into the **Master** branch. This should
    only be used sparingly and when necessary and should not be used to circumvent proper peer-review and testing
    processes.

- Release:
    When the develop branch is sufficiently mature, a **Release** branch is branched off in order to prepare the
    codebase for release. Such tasks include updating documentation, further code-review and
    integration/acceptance testing.


Add your ProcessTask to production
----------------------------------

1. Pull request on Bitbucket

Once your feature branch is complete, make sure that your changes have been committed (you should be committing all
the way!). Then online on Bitbucket you need to open a Pull Request to merge the branch to ``develop``.

This will also trigger a review process. Once the reviewers are satisfied and all the tests are passing your
Pull Request can be accepted and your changes added to the codebase.

Once the develop branch is ready a release branch can be created from the develop branch in order to prepare for a new
release!

2. Release branch

- Update version in hazenlib/__init__.py
- Update documentation: ``sphinx-build docs/ docs/_build``
- Ensure build is passing
- Merge with ``master`` branch

3. Master branch

- Log into Heroku, to view ``staging`` and ``production`` apps
- Run user acceptance tests on hazen-staging
- Promote hazen-staging to production!


