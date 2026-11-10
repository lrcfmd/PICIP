# ---- builder ----
FROM ghcr.io/astral-sh/uv:debian-slim AS builder

ENV UV_PYTHON_INSTALL_DIR=/opt/uv
ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.12

WORKDIR /app

COPY pyproject.toml uv.lock ./

# --compile-bytecode: ship .pyc files, so each run (a fresh container, as
# uid 1000 that cannot write them back) does not recompile every import.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable --compile-bytecode

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --compile-bytecode


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
