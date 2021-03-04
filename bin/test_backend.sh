#!/usr/bin/env bash

# Exit if any subcommand fails
set -e

# Install poetry for your system: https://python-poetry.org/docs/#installation
poetry --version || curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Install the package with poetry
poetry install

# Run the tests
poetry run pytest .