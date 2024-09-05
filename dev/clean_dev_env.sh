#!/bin/bash

# This script ensures all local Nomadic build environment logic is
# properly shutdown and cleaned.

set -ex

poetry env remove --all
