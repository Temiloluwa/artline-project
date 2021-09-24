## Artline Project

An Image Processing app that deploys the model at <https://github.com/vijishmadhavan/ArtLine>

- Model is deployed using Flask on an Azure VM that's offline attimes due to cost savings
- Images supplied as queries to the model are queued on the server using Redis and Redis queue (RQ)