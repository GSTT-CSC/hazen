FROM python:3.9
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN groupadd -r hazen_user && useradd --create-home --shell /bin/bash -r -g hazen_user hazen_user
COPY . /home/hazen_user/hazen
WORKDIR /home/hazen_user/hazen
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py install
USER hazen_user
WORKDIR ../data
ENTRYPOINT ["hazen"]

# docker build -t docker_hazen .
## run interactive
# docker run -it --rm --name hazen --mount type=bind,source="$(pwd)",target=/home/hazen_user/data docker_hazen /bin/bash
#docker run -it --entrypoint /bin/bash --mount type=bind,source="$(pwd)",target=/home/hazen_user/data gsttmriphysics/hazen
## run normal
# docker run --rm --mount type=bind,source="$(pwd)",target=/home/hazen_user/data -w /home/hazen_user/data docker_hazen