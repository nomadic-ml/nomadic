#!/bin/bash

# This script ensures we run all Nomadic unit tests

set -ex

## Run tests

poetry run pytest tests/unit --cov=nomadic
poetry run coverage report
