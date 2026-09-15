FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-runtime.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-runtime.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 moderator \
    && mkdir -p /data/incoming /data/reports /moderation-scratch \
    && chown -R moderator:moderator /data /moderation-scratch

USER moderator
EXPOSE 8088
CMD ["uvicorn", "reel_moderation.service:app", "--host", "0.0.0.0", "--port", "8088"]
