#!/usr/bin/env bash

# Exit if any subcommand fails
set -e

bold=$(tput bold)
normal=$(tput sgr0)
underline=$(tput sgr 0 1)

echo "${bold}==== Setting up the backend =================================================${normal}"

bash ./bin/setup_and_start_docker.sh
bash ./bin/clone_backend.sh

# Build an image from the provided Dockerfile
docker build -t athology-ml ./athology-ml

# Create a container (if it does not already exist) and run the backend
docker start athology-ml || docker run -d --name athology-ml -p=5000:5000 athology-ml

# Health-check on the backend
echo "Performing health-check on the backend..."
sleep 1  # sleep for a second to give the backend a chance to start
curl -X GET "http://localhost:80/"
printf "\n"

echo "${bold}Backend is running at ${underline}http://localhost:80/${normal}. See ${underline}http://localhost:80/docs${normal} for details.${normal}"
echo "${bold}Run docker stop athology-ml to stop it${normal}"