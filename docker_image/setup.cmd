# Creation des images

docker image build ./chpy_api -t bbastien1/chpy_api:latest -t bbastien1/chpy_api:1.0.2


docker image push bbastien1/chpy_api:1.0.2
docker image push bbastien1/chpy_api:latest


# Docker-Compose
docker-compose up