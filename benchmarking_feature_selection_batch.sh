#!/bin/bash

# Set script variables
VENV_PATH="/home/dawsonlan/ode-biomarker-project/.venv/bin/activate"
PYTHON_SCRIPT="/home/dawsonlan/ode-biomarker-project/benchmarking_feature_selection_batch.py"
PROJECT_DIR="/home/dawsonlan/ode-biomarker-project"

# Change to project directory
cd "$PROJECT_DIR" || { echo "Failed to change to project directory"; exit 1; }

# Check if virtual environment exists
if [ ! -f "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Source the virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH"

# Run the Python script
echo "Running benchmarking script..."
python "$PYTHON_SCRIPT"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Script completed successfully"
else
    echo "Script failed with error code $?"
    exit 1
fi