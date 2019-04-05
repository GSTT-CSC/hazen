# How to contribute

- Start by creating a new issue for the feature
- Create a Git feature-branch for this ticket
- *Coding magic happens here*
- Keep an eye on the feedback from Bitbucket Pielines and Sonarcube
- Check a running instance of your app by locally running the latest Docker image:docker run --rm -p5000:5000 -it
- If you’re satisfied with the new feature, open a Pull Request.


### Sonarcube
To run local, install sonar-scanner. Edit properties file in conf so that url is pointing to cloud instance.

## Branching
- Git-flow.

## Merging

- Pull request. 
- Haris is final reviewer.

## Releasing

- Produce requirements.txt: __pipreqs --force --savepath ./requirements.txt --ignore bin,hazen-venv ./__
- Make sure all tests are passing
- Update version in hazen/\_\_init\_\_.py, remove 'dev'.
- Update and build docs



