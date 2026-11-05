#!/usr/bin/env bash
# Starts Focus Flow on http://localhost:5000
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt && python3 app.py
