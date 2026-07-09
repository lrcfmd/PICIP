# ---- builder ----
FROM ghcr.io/astral-sh/uv:debian-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv python install

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable


# ---- runtime ----
FROM debian:trixie-slim

WORKDIR /app

COPY --from=builder /root/.local/share/uv /root/.local/share/uv
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /data/output

ENTRYPOINT ["model-run"]
