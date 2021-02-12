# Athology-ML

This repository contains our back-end, which exposes several REST API endpoints for computation and prediction on accelerometer data.

## Installation

This repository requires Python 3.8 or later. Installation is managed with [Poetry](https://python-poetry.org/).

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/dcsil/athology-ml
cd athology_ml

# Install the package with poetry
poetry install
```

## Usage

To start the web service, run

```bash
uvicorn athology_ml.main:app \
    --host 0.0.0.0 \
    --port 5000 \
    --reload
```

For a health check on a running service, run

```
curl -X GET "http://localhost:5000/" 
```

For documentation, visit [`http://localhost:5000/docs`](http://localhost:5000/docs) in your browser.