# From the base image
# here: https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker
# Based off the example
# here: https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker#dependencies-and-packages
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy using poetry.lock* in case it doesn't exist yet
COPY ./pyproject.toml ./poetry.lock* /app/

RUN poetry install --no-root --no-dev

COPY ./athology_ml/ /app/athology_ml