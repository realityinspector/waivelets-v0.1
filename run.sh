#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null
python -m shiny run app.py --port "${1:-8765}"
