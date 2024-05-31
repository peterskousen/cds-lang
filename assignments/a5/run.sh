#!/usr/bin/bash

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
clear
python src/plot_emissions.py
deactivate