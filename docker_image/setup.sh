# Recuperation de la derniere version des fichiers
rmdir -rf docker_image/chpy_api/app
cp -r ../app/ docker_image/chpy_api/app/

# Creation des images
docker image build ./chpy_api -t chpy_api:latest

# Docker-Compose
docker-compose up