## Artline Project

An Artistic Image Generator app for the model @ <https://github.com/vijishmadhavan/ArtLine>

- Web App link: http://20.193.237.75

## Technlogy stack
Model runs on celery task queues. It receives and returns an image to the web app as a base64 string.

- Front-End Library: Boostrap
- Web Server: UWSGI
- Web Framework: Flask
- Task Queue: Celery
- Broker and Result Backend - Redis
- Model: Pytorch and Fast.AI
- Deployment Platform - Azure

## Availability
Due to cost savings, model is online daily between 12:00 (CET) and 18:00 (CET)