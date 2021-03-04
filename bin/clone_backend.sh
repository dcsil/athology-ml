#!/usr/bin/env bash

# Exit if any subcommand fails
set -e

bold=$(tput bold)
normal=$(tput sgr0)

echo "${bold}==== Cloning the backend ====================================================${normal}"

# Clone the backend repo, if it doesn't already exist
if [ ! -d athology-ml ] ; then
    git clone https://github.com/dcsil/athology-ml
fi

# Make sure the repo is up to date
cd athology-ml
git checkout main
git pull
cd ../