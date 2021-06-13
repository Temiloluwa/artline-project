FROM python:3.8-buster

WORKDIR /app
COPY app/app.ini .

COPY . .

RUN pip install -r requirements.txt

CMD ["uwsgi", "app.ini"]