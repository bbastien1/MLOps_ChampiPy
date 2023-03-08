FROM alpine:latest
RUN apk add --no-cache python3 py3-pip

RUN pip install fastapi==0.93.0
RUN pip install uvicorn==0.20.0

ADD main.py /chpy_api/main.py

WORKDIR /chpy_api/

EXPOSE 8000

CMD uvicorn main:api --reload
