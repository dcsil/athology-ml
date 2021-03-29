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

RUN poetry install --no-root --no-dev --extras "production"

COPY ./athology_ml/ /app/athology_ml

# Because our main.py lives somewhere other than the default, we need to tell FastAPI where.
# See: https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker#module_name
ENV MODULE_NAME="athology_ml.app.main"
# This is a hack to try to fit within Heroku's RAM limits. Each worker will consume some amount
# of RAM, so my thinking is to restrict the FastAPI image to a single worker to use the least
# amount of RAM.
ENV MAX_WORKERS="1"