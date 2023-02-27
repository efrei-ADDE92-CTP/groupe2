FROM python:3.8-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 80

CMD flask run -h 0.0.0.0 -p 80