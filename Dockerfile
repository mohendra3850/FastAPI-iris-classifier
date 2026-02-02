FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
COPY app.py .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80" ]

LABEL Name = "fastapi-model-api"
LABEL Version = "1.0.0"

