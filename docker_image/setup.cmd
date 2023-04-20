:: Recuperation de la derniere version des fichiers
rmdir /S /Q docker_image\chpy_api\app
xcopy /E ..\app\ docker_image\chpy_api\app\

:: Creation des images
docker image build ./chpy_api -t chpy_api:latest -t

:: Creation + push
::docker image build ./chpy_api -t bbastien1/chpy_api:latest -t bbastien1/chpy_api:1.0.2
::docker image push bbastien1/chpy_api:1.0.2
::docker image push bbastien1/chpy_api:latest

:: Docker-Compose
docker-compose up