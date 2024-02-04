from setuptools import setup, find_packages

setup(
    name="spanet",
    packages=find_packages(),
    version="0.1.0",
    description="All-In-One Topology-Aware Particles Transformer Encoder",
    author="Jovan Mitić",
    author_email="jovan.mitic@cern.ch",
    url="https://github.com/Joxy97/AI1-TAPTE",
    install_requires=[
        'numpy',
        'h5py',
        'lightning',
        'tensorboard',
        'scikit-learn',
        'itertools',
        'tqdm',
        'prettytable',
        'awkward',
        'click',
        'uproot',
        'vector',
        'coffea',
        'pathlib',
    ],
)
