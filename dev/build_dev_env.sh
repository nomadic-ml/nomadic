#!/bin/bash

# This script builds the global Nomadic build environment on MacOS/Linux.

set -ex

# ----------------- Constants Start --------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ----------------- Constants End ----------------------------

# ----------------- Functions Start --------------------------

# Function to install a specific Python version if not already installed
function install_python_version {
  local version=$1

  if pyenv versions --bare | grep -q "^${version}$"; then
    echo "Python ${version} is already installed. Skipping installation."
  else
    echo "Python ${version} is not installed. Installing..."
    pyenv install ${version}
  fi
}

# ----------------- Functions End ----------------------------

# Install Homebrew, if not exists.
if which -s brew; then
    echo "Homebrew is already installed."
else
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || true
    (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> ~/.zshrc
    eval "$(/opt/homebrew/bin/brew shellenv)"
    source ~/.zshrc
fi
brew update

# Install pyenv & pipx, if not exists
brew install pyenv pyenv-virtualenv pipx
# Install Python $PYTHON_VERSION
install_python_version $PYTHON_VERSION
# Use Python $PYTHON_VERSION
pyenv local $PYTHON_VERSION
pyenv version
source ~/.zshrc

# Install pipx
pipx ensurepath
source ~/.zshrc

# Make .venv from current PyEnv Python
pyenv exec python -m venv .venv

source $ROOT_DIR/.venv/bin/activate

# Install Poetry via pipx
pipx install poetry

# Enable Poetry making .envs in Nomadic project folder
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true

# Install Nomadic and all other required packages for development
poetry install

# Configure pre-commit
poetry run pre-commit install
