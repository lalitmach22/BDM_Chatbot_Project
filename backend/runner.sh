#!/bin/bash

# Stop script on error
set -e

# Check if Python venv exists, if not create it
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed, and install if necessary
echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run the main script
echo "Running main.py..."
python3 main.py
