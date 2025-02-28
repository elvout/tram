#! /usr/bin/env bash

set -e

[[ ! -d .venv ]] && python3 -m venv .venv

source .venv/bin/activate

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install 'git+https://github.com/facebookresearch/detectron2.git@a59f05630a8f205756064244bf5beb8661f96180'
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
# conda install -c conda-forge suitesparse

pip install pulp
pip install supervision

pip install open3d
pip install opencv-python
pip install loguru
pip install git+https://github.com/mattloper/chumpy
pip install einops
pip install plyfile
pip install pyrender
pip install segment_anything
pip install scikit-image
pip install smplx
pip install timm==0.6.7
pip install evo
pip install pytorch-minimize
pip install imageio[ffmpeg]
pip install numpy==1.26.4
pip install gdown
pip install openpyxl
# pip install git+https://github.com/princeton-vl/lietorch.git

# chumpy's __init__.py attempts to import types from numpy that were removed in
# numpy 1.24, which causes an ImportError for newer versions of numpy (which are
# in turn required by various other packages). There does not appear to be a
# reason for the import statement. chumpy appears to no longer be maintained, so
# we'll manually remove the offending import statement.
perl -0777 -i -p -e "s/from numpy import.+?\n//" .venv/lib/python3.10/site-packages/chumpy/__init__.py
