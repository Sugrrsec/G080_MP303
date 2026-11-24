
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY docker-requirements.txt .

RUN pip install --no-cache-dir -r docker-requirements.txt

COPY src ./src
COPY model ./model

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "mp303.api:app", "--host", "0.0.0.0", "--port", "8000"]
