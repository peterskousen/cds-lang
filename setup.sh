#!/usr/bin/bash

# create virtual env
python -m venv .venv

#activate env
source ./.venv/bin/activate

# requirements
pip install --upgrade pip
pip install -r requirements.txt

deactivate