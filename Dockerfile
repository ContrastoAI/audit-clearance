FROM python:3.12-slim

RUN apt-get update \ 
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    make

WORKDIR /app

COPY arena ./arena
COPY main.py ./main.py
COPY Makefile ./Makefile
COPY task_pairs ./task_pairs
COPY .env ./.env
COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
COPY README.md ./README.md

COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /uvx /bin/

RUN uv sync --locked --no-dev --no-install-project

CMD ["make", "run"]
