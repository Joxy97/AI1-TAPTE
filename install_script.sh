#!/bin/bash

python -m pip install --user virtualenv
virtualenv ai1_tapte_env
source ai1_tapte_env/bin/activate

pip3 install -e .
pip3 install torch torchvision torchaudio
