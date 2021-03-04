![build](https://github.com/dcsil/athology-ml/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/dcsil/athology-ml/branch/main/graph/badge.svg?token=1g0QJXSYmo)](https://codecov.io/gh/dcsil/athology-ml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub](https://img.shields.io/github/license/dcsil/athology-ml?color=blue)

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
uvicorn app.main:app \
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
docker run -d --name athology-ml -p 80:80 athology-ml

# Health check
curl -X GET "http://localhost:80/" 
```

Once the container has been created, you can stop/start it with

```bash
docker stop athology-ml
docker start athology-ml
```

### Tuning your own Model

If you want to tune your own model, please install with

```
poetry install -E tune
```

You can tune a model with the `tune` command. Call

```
tune --help
```

for usage details. In short, you will minimally need to a path to a `directory` structured like

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

Where each CSV file is expected to contain accelerometer data that will be used for training, testing and validation.