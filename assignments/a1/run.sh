#!/usr/bin/bash

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_md
clear -x
python src/feature_extraction.py
deactivate