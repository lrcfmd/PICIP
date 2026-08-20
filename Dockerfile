# ---- builder ----
FROM ghcr.io/astral-sh/uv:debian-slim AS builder

ENV UV_PYTHON_INSTALL_DIR=/opt/uv
ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.12

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable


# ---- runtime ----
FROM debian:trixie-slim

RUN groupadd -r lmds \
    && useradd -r -g lmds -u 1000 -m -d /home/lmds lmds

WORKDIR /app

COPY --from=builder /opt/uv /opt/uv
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /data/output \
    && chown -R 1000:1000 /app /data/output /home/lmds

USER 1000:1000

ENTRYPOINT ["model-run"]
