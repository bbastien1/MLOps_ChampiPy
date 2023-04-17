#FROM python:latest
FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

WORKDIR /chpy_api/
ADD /app/ /chpy_api/

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD uvicorn api.main:api --reload
