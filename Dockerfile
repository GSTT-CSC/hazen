FROM python:3.7
ADD requirements.txt /requirements.txt
ADD hazen.py /hazen.py
ADD app /app
ADD hazenlib /hazenlib
RUN pip install -r /requirements.txt
ENTRYPOINT ["python"]
CMD ["./hazen.py","--host=0.0.0.0"]
#ENTRYPOINT celery -A test_celery worker --concurrency=20 --loglevel=info

