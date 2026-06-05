#!/bin/bash
# Adeptus Mechanicus Deployment Script
set -e

echo "=== Activating Deployment Protocols ==="
PROJECT_DIR="${SERVO_SKULL_HOME:-/opt/servo-skull}"
cd "$PROJECT_DIR"

echo "Step 1: Pulling latest commits from repository..."
git pull

echo "Step 2: Syncing dependencies..."
uv sync --reinstall

echo "Step 3: Restarting servo-skull-worker systemd service..."
sudo systemctl restart servo-skull-worker

echo "=== Deployment Completed Successfully ==="
