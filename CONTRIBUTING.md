## Introduction 
The documentation is for people intending to contribute code to the Hazen project. Adapted from ull guidelines, found at https://gsttmri.atlassian.net/wiki/spaces/HAZEN/pages/37454040/Contribute+new+code .  

## How to contribute

1. All contributions must be linked to an issue. Issues are organised on Jira (https://gsttmri.atlassian.net/secure/RapidBoard.jspa?rapidView=5&view=planning&selectedIssue=HZN-48&issueLimit=100). 
    
    If your issue already exists, but you think it is not completely solved or needs more information, please add more details as a comment.

    If your issue doesn't already exist:
    - Gather any user research
    - Define acceptance test criteria
    - Make your issue! Name the issue in the format {component}-{issue summary}, e.g. SNR-Fix ROI size for large FOV, try to use the imperative mood where possible..  
      Be as detailed as possible with the description, including detailed steps on how to reproduce the issue if it is a bug. Please offer a resolution in the description if you feel you are able to.
 
2. Review implementation plan with Haris, who will assign the issue to you if there are no overlaps

3 Create a new issue specific feature-branch for this issue, in the format {component}-{issue summary}. You can do this through the issue cards on Jira. 
    - Click on the issue card and detailed view should open on the right. Scroll to the bottom and under the development section, select the ‘Create branch’.
    - This will open a new page, make sure you’ve selected the correct repository gsttmri/hazen
    - Select type of branch, this will almost always be ‘Feature’ - if you are not sure speak to Haris.
    - The From branch for ‘Feature’ branches is always ‘develop’ - again, if you are unsure speak to Haris.
    - The branch name will be auto-filled, try to use this one as much as possible. If the name is cropped, edit the ending so that the branch name is clear.
    - Hit create!

4. Check out your new branch in source tree, a copy of your branch should now appear in your local branches

5. Create a Pull Request on Bitbucket, requesting a (future!) merge of your branch back to ‘develop’.
    - Use all the auto-filled details, except please add WIP- to the front of the title e.g. WIP-Feature/77 relaxometry, this way the maintainer knows you are still working (Work In Progress) and not ready for final review. 
    - Your branch will eventually be merged with the codebase via a Pull Request. This allows your code to be formally reviewed and approved. 

5. Make your changes to the branch! Always start with unit tests. 

5. Check that all tests are still passing, and that your change is covered by the tests. Tests can be run in the terminal using the command
   "pytest tests"

7. Ensure tests are passing on Bitbucket Pipelines 

6. Create pull request from the task branch to develop, making sure every commit message includes the issue code e.g. HZN-41 (more on commit message convention below). When you are ready for your code to be formally reviewed, remove ‘WIP-' from the PR title, and ping the maintainer a message.

8. If feedback is received, make the changes and keep a dialog open via bitbucket

9. If there are conflicts between the pull request branch and the develop branch, pull the changes from the develop branch and resolve the conflicts locally.

10. The process finishes when the branch is merged with the develop branch 


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

- Haris will create a new ‘Release’ branch from develop
- Haris, as maintainer, is responsible for releasing new version of code
- Any edits or documentation updates will be made in the ‘Release’ branch
- Finally, the ‘Release’ branch is merged with ‘master’ and a new version is tagged for release!


