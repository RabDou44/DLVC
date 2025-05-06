#!/bin/bash

# Unpack Script: Unzips backup, creates Python virtual environment, and installs requirements

# === CONFIGURATION ===
BACKUP_FILE="$1"  # First argument should be the path to the ZIP file
TARGET_DIR="./unpacked_backup"
VENV_DIR="$TARGET_DIR/venv"

# === UNZIP ===
echo "Unpacking $BACKUP_FILE to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"
unzip -q "$BACKUP_FILE" -d "$TARGET_DIR"

if [ $? -ne 0 ]; then
  echo "Unzipping failed. Check if the ZIP file exists and is valid."
  exit 1
fi

echo "Unzip completed."

# === SETUP VIRTUAL ENVIRONMENT ===
if [ -f "$TARGET_DIR/requirements.txt" ]; then
  echo "Setting up Python virtual environment..."
  python3 -m venv "$VENV_DIR"

  if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Make sure python3 and venv are installed."
    exit 2
  fi

  source "$VENV_DIR/bin/activate"
  echo "Installing dependencies from requirements.txt..."
  pip install --upgrade pip
  pip install -r "$TARGET_DIR/requirements.txt"

  if [ $? -eq 0 ]; then
    echo "Environment setup complete and requirements installed."
  else
    echo "Failed to install requirements."
    exit 3
  fi
else
  echo "No requirements.txt found in unpacked files. Skipping virtual environment setup."
fi

echo "Unpacking and environment setup done."
