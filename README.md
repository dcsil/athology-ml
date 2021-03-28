![build](https://github.com/dcsil/athology-ml/workflows/build/badge.svg)
[![Maintainability](https://api.codeclimate.com/v1/badges/239f9b50087ac4bd1df9/maintainability)](https://codeclimate.com/repos/6025b89e157a2d0162008939/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/239f9b50087ac4bd1df9/test_coverage)](https://codeclimate.com/repos/6025b89e157a2d0162008939/test_coverage)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

# Athology-ML

This repository contains our back-end, which exposes several REST API endpoints for computation and prediction on accelerometer data.

## Installation

This repository requires Python 3.7 or later. Installation is managed with [Poetry](https://python-poetry.org/).

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/dcsil/athology-ml
cd athology-ml

# Install the package with poetry
poetry install
```

## Usage

To start the web service, run

```bash
uvicorn athology_ml.app.main:app \
    --host 0.0.0.0 \
    --port 80 \
    --reload
```

For a health check, run

```
curl -X GET "http://localhost:80/" 
```

For documentation, visit [`http://localhost:80/docs`](http://localhost:80/docs) in your browser.

### Deploying with Docker

To deploy with Docker, first build an image from the provided Dockerfile

```bash
docker build -t athology-ml .  
```

Then you can create a container and run the web service with

```bash
docker run -d --name athology-ml -p 80:80 -e PORT="80" athology-ml
```

Once the container has been created, you can stop/start it with

```bash
docker stop athology-ml
docker start athology-ml
```

### Train and Tune your own Model

If you want to train and tune your own model, please install with

```
poetry install -E train
```

Each model has its own subcommand. To see each subcommand, call

```bash
athology-ml --help
```

For example, to train and tune a `jump-detection` model, call

```
athology-ml jump-detection train path/to/dataset
```

For details on the arguments and options of any subcommand, invoke them with `--help`

```
athology-ml jump-detection train --help
```

In this case, we minimally need a path to a `directory` with a dataset of CSV files, structured like

```
.
├── train
│   ├── 20191006_rider0_Accelerometer_Manualtagged.csv
│   └── 20200108_rider1_Accelerometer_Manualtagged.csv
├── valid
│   └── 20200108_rider3_Accelerometer_Manualtagged.csv
└── test
    ├── 20191006_rider4_Accelerometer_Manualtagged.csv
    └── 20200106_rider5_Accelerometer_Manualtagged.csv
```

Where each CSV is expected to contain the columns `"x-axis (g)"`, `"y-axis (g)"`, `"z-axis (g)"`, and `"is_air"`.
