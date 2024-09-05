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

# Ensure Node.js is installed
./dev/install_nodejs.sh

# Install Docker & Colima
brew install docker
brew install colima

# Start Colima
colima start

# Check if Docker is running
if is_docker_running; then
    echo "Docker is already running."
else
    start_docker
fi

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

# Install git-lfs with Homebrew
brew install git-lfs
git lfs install

# Install Supabase, if not exists.
if supabase --version; then
    which -s supabase
else
    # To-Do: Fix below, remove the || true that silenty catches supabase installation failure
    brew install supabase/tap/supabase || true
fi

# Start supabase
supabase stop --no-backup
supabase start
if [[ $? != 0 ]] ; then
    # If installation fails for the first time, this is most likely
    # due to a supabase bug. Relevant link: https://github.com/supabase/cli/issues/1938
    # Apply below fix and re-attempt supabase installation.
    echo -n "v2.142.0" > $ROOT_DIR/supabase/.temp/gotrue-version
    echo -n "v12.0.1" > $ROOT_DIR/supabase/.temp/rest-version
    supabase stop --no-backup
    supabase start
fi

# Install pyenv, if not exists
brew install pyenv pyenv-virtualenv
# Install Python $PYTHON_VERSION
install_python_version $PYTHON_VERSION
# Use Python $PYTHON_VERSION
pyenv local $PYTHON_VERSION
pyenv version
source ~/.zshrc


# Install pipx
brew install pipx
pipx ensurepath
source ~/.zshrc
# sudo pipx ensurepath -global # optional to allow pipx actions in global scope. See "Global installation" section below.

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
