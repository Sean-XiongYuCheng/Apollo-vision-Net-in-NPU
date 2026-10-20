#!/bin/bash
set -e

pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install --no-deps lerobot==0.4.0

echo "All dependencies installed successfully!"
