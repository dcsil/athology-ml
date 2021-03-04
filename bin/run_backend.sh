#!/usr/bin/env bash

# Exit if any subcommand fails
set -e

# This is set by the FastAPI docker image
PORT=80

echo "======= Setting up the backend =============================================================="

bash ./bin/setup_and_start_docker.sh

# Build an image from the provided Dockerfile
docker build -t athology-ml .

# Create a container (if it does not already exist) and run the backend
docker start athology-ml || docker run -d --name athology-ml -p=80:80 athology-ml

# Health-check on the backend
echo "Performing health-check on the backend..."
sleep 1  # sleep for a second to give the backend a chance to start
curl -X GET "http://localhost:${PORT}/"
printf "\n"

echo "Backend is running at http://localhost:${PORT}/. See http://localhost:${PORT}/docs for details."
echo "Run docker stop athology-ml to stop it"