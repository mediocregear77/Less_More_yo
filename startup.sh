#!/bin/bash
# ------------------------------------------
# Nexi_One Startup Script
# ------------------------------------------

# Display a starting message
echo "Starting Nexi_One initialization..."

# Activate the virtual environment
# Check if the 'venv' directory exists
if [ -d "venv" ]; then
  # Source the activation script for the virtual environment
  source venv/bin/activate
  echo "Virtual environment activated."
else
  # If the venv directory is not found, print an error and exit
  echo "Error: Virtual environment not found. Please run the setup script first."
  exit 1
fi

# Set Python to run in unbuffered mode, which is good for logging and output
export PYTHONUNBUFFERED=1

# Launch the Nexi system's initializer
# The init.py script is now responsible for all core module initialization
echo "Launching Nexi_One core..."
python3 deployment/init.py

# Exit with the return code of the Python script
# This ensures that the shell script's exit status reflects the success or failure of the Python program
exit $?
