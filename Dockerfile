#FROM python:3.7
#ADD requirements.txt /requirements.txt
#ADD hazen.py /hazen.py
#ADD app /app
#ADD hazenlib /hazenlib
#RUN pip install --no-cache-dir -r /requirements.txt
#ENTRYPOINT ["python"]
#CMD ["./hazen.py","--host=0.0.0.0"]
##ENTRYPOINT celery -A test_celery worker --concurrency=20 --loglevel=info
#RUN useradd --create-home --shell /bin/bash hazen_user

# this version works but uses root user permissions
#FROM python:3.6
#COPY requirements.txt ./
#RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r requirements.txt
#COPY . .
#CMD ["bash"]
# docker build -t docker_hazen .
# docker run -it --rm --mount type=bind,source=`pwd`,target=. docker_hazen /bin/bash


# this version is wip, trying to set up for non-root user
FROM python:3.6
#RUN useradd --create-home --shell /bin/bash hazen_user

RUN groupadd -r hazen_user && useradd --create-home --shell /bin/bash -r -g hazen_user hazen_user

COPY . /home/hazen_user/hazen
WORKDIR /home/hazen_user/hazen
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
#COPY . /home/hazen_user/hazen
RUN python setup.py install
WORKDIR ../data
USER hazen_user
#CMD ["bash"]
ENTRYPOINT ["hazen"]

# docker build -t docker_hazen_2:hazen .
# cd /Users/lj16/King\'s\ College\ London/GSTT\ MRI\ Physics\ -\ Shared\ Files\ -\ Shared\ Files/quality/qa/BMI\ Sites/BMI_Chaucer_15T_GE_Signa/2020/data/
## run interactive
# docker run -it --rm --mount type=bind,source=`pwd`,target=/home/hazen_user/data docker_hazen_2 /bin/bash
## run normal
# docker run --rm --mount type=bind,source=`pwd`,target=/home/hazen_user/data -w /home/hazen_user/data docker_hazen_2