FROM python:3.9
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Enable Adjustable Permissions
# Note: these values can be overridden during docker build, e.g.:
# docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f Dockerfile -t hazen_test_img .
ARG UNAME=hazen_user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID --create-home --shell /bin/bash -r $UNAME

COPY . /home/$UNAME/hazen

# Python Install
WORKDIR /home/$UNAME/hazen
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py install

# Final Setup
WORKDIR /home/$UNAME/data
USER $UNAME
ENTRYPOINT ["hazen"]

# docker build -t docker_hazen .
## run interactive
# docker run -it --rm --name hazen --mount type=bind,source="$(pwd)",target=/home/hazen_user/data docker_hazen /bin/bash
# docker run -it --entrypoint /bin/bash --mount type=bind,source="$(pwd)",target=/home/hazen_user/data gsttmriphysics/hazen
## run normal
# docker run --rm --mount type=bind,source="$(pwd)",target=/home/hazen_user/data -w /home/hazen_user/data docker_hazen