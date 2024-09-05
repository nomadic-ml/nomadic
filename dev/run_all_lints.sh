#!/bin/bash

# This script ensures we run all Nomadic lints

set -ex

source .venv/bin/activate
## Run lints
# stop the build if there are Python syntax errors or undefined names
poetry run flake8 nomadic/ --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
poetry run flake8 nomadic/ --count --exit-zero --max-complexity=12 --max-line-length=127 --statistics
