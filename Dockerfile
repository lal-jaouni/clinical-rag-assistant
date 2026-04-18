FROM python:3.12-slim AS base

WORKDIR /app

# System deps for psycopg (PostgreSQL client) and pdfplumber
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["python", "-m", "api.app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
