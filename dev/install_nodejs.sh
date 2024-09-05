#!/bin/bash

# This script install Node.js and its requirements.
# Requirements:
#   - Docker Desktop: Install from: https://docs.docker.com/desktop/
#       - Ensure that Docker Desktop is also running.

set -ex

# installs nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Source nvm directly
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# download and install Node.js (you may need to restart the terminal)
nvm install 22
# verifies the right Node.js version is in the environment
node -v # should print `v22.3.0`
# verifies the right NPM version is in the environment
npm -v # should print `10.8.1`

# Install required npm packages
npm i -g next@latest react@latest react-dom@latest vercel@latest mintlify@latest concurrently@latest react-plotlyjs@latest
