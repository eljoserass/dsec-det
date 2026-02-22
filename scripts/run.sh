#!/bin/bash
# Run the upload script in the background, detached from the terminal.
# Survives SSH disconnect. Check progress.txt for status.

cd "$(dirname "$0")"
# python3.13 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
nohup python3 dsec_to_r2.py > output.log 2>&1 &
echo "PID: $!"
echo "Check progress:  cat scripts/progress.txt"
echo "Check full log:  tail -f scripts/output.log"
