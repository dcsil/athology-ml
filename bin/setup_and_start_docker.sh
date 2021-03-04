#!/usr/bin/env bash

# Exit if any subcommand fails
set -e

bold=$(tput bold)
normal=$(tput sgr0)

echo "${bold}==== Setting up Docker ======================================================${normal}"

# Install Docker using homebrew (if not already installed)
if (! docker --version); then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew --version || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install docker
    else
        if (! brew --version); then
            test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
            test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
            test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
            echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
            brew install docker
        fi
    fi
fi
# Check if Docker is running, if not, try to start it. This should work on MacOS and Linux
# https://stackoverflow.com/a/48843074/6578628
if (! docker stats --no-stream > /dev/null 2>&1); then
    echo "Docker is not running. Attempting to start it for you..."
    open /Applications/Docker.app > /dev/null 2>&1 || sudo systemctl start docker > /dev/null 2>&1 || sudo service docker start > /dev/null 2>&1
while (! docker stats --no-stream > /dev/null 2>&1); do
    sleep 1
done
fi