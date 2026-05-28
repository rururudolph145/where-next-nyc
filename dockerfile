FROM python:3.11

WORKDIR /workspace

COPY app/ app/
COPY models/ models/
COPY config.yml .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "80"]
