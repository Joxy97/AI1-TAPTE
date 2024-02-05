#!/bin/bash

python3 -m pip install --user virtualenv
virtualenv ai1_tapte_env
source ai1_tapte_env/bin/activate

python3 -m pip install --upgrade pip
pip3 install -e .
pip3 install torch torchvision torchaudio
