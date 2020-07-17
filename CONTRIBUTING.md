## Introduction 
The documentation is for people/instituions intending to contribute towards Hazen. 

## How to contribute

1. Start by creating a new issue on bitbucket for the feature and:
    - decide which release this is for
    - gather any user research
    - define acceptance test criteria
2. Check if there is already a branch dedicated to the issue, if not, create a new Git feature-branch for this ticket, named [issue number] issue name] .  
3. Make your changes to the branch
4. Check that all tests are still passing, and that your change is covered by the tests
5. Create pull request from the task branch to master with a detailed commit message (see next section for commit convention) 
6. Keep an eye on the feedback from Bitbucket Pipelines 
7. If feedback is received, make the changes and keep a dialog open via bitbucket
8. If there are conflicts between the pull request branch and the master branch, pull the changes from the master and resolve the conflicts locally.
9. The process finishes when the branch is merged with the master branch 


## Commit Message Convention 
Concise and clear commit messages help to keep track of why a certain change to the code has been made. 
Please use the following template when making your commit messages: 

[SUMMARY LINE] Issue Number, summarise change in 50 characters or less in the imperative mood
 
[BODY OF COMMIT MESSAGE] 

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical (unless
you omit the body entirely); various tools like `log`, `shortlog`
and `rebase` can get confused if you run the two together.
 
Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.
 
Further paragraphs come after blank lines.
 
 - Bullet points are okay, too
 
 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here
 
If you use an issue tracker, put references to them at the bottom,
like this:
 
Resolves: #123 [1]

See also: #456 [2], #789 [3]

[1](http:// some url)

[2](http:// some url)

[3](http:// some url)


## Unit Testing 
Hazen tests are located under tests/. •	The unit test's file name follows test_[module_name].py.


## Changing DB models
Remember to to generate migration script by running: __flask db migrate -m "very short commit message"__

## Sonarcube
To run local, install sonar-scanner. Edit properties file in conf so that url is pointing to cloud instance.

## The Code reviewing process

- After your pull request has been submitted, Haris is the final reviewer.

## Releasing

- Produce requirements.txt: __pipreqs --force --savepath ./requirements.txt --ignore bin,hazen-venv ./__
- Make sure all tests are passing
- Update version in hazenlib/\_\_init\_\_.py, remove 'dev'.
- Update docs (try sphinx-apidoc for autodocumentation of modules)


