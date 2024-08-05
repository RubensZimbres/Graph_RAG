FROM python:3.10

EXPOSE 8080
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python3","/app/src/app.py", "--server.port=8080", "--server.address=0.0.0.0"]