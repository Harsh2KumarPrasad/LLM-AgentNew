FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

RUN pip install fastapi pillow uvicorn markdown duckdb requests httpx pytesseract sentence_transformers
# Ensure the installed binary is on the 'PATH'
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
RUN mkdir -p /data
COPY main.py /app

EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
