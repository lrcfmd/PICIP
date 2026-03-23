FROM python:3.12-slim

WORKDIR /app
COPY dist/*.whl /app/
RUN pip install /app/*.whl

RUN mkdir -p /data/output

ENTRYPOINT ["picip-run"]
