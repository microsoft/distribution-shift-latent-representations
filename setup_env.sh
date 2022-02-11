# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Create conda env.
conda create -n dist_shift python=3.8 -y;
conda activate dist_shift;
conda install -c anaconda ipykernel -y

# Install ipykernel on aml machine:
python -m ipykernel install --user --name dist_shift

# Helper packages
pip install bidict
pip install faiss-cpu
pip install networkx
pip install opencv-python
pip install tqdm
pip install scikit-learn
pip install scipy
pip install pot
pip install statsmodels

# Install TDA packages
pip install scikit-tda
pip install umap-learn

# Install torch
pip install torch

# Install using local setup.py.
pip install -e .

