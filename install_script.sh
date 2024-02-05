#!/bin/bash

python3 -m pip install --user virtualenv
virtualenv ai1_tapte_env
source ai1_tapte_env/bin/activate

python3 -m pip install --upgrade pip
pip3 install numpy
pip3 install h5py
pip3 install torch torchvision torchaudio
pip3 install lightning
pip3 install tensorboard
pip3 install scikit-learn
pip3 install more-itertools
pip3 install tqdm
pip3 install prettytable
pip3 install awkward
pip3 install click
pip3 install uproot
pip3 install vector
pip3 install coffea
pip3 install pathlib
