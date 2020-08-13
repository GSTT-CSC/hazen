FROM python:3.6
ENV FLASK_APP=hazen.py
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install pip --upgrade
RUN pip install -r requirements.txt
COPY . /code/