# VFC_VCM (Minimal TransVFC-like Pipeline)

This is a minimal, runnable PyTorch project that implements a simplified TransVFC-like video feature compression pipeline using dummy data. It uses local `compressai` (expected to be available in the workspace root `compressai/`) and wraps a `bmshj2018_hyperprior` model for feature/motion/residual codecs.

Features:
- Feature extraction, motion estimation, motion compression, compensation, residual compression, reconstruction.
- Rate-distortion loss (MSE + lambda * rate).
- Dummy dataset producing random image pairs.
- Simple trainer that runs a small number of iterations.

Requirements
------------
- Python 3.8+
- PyTorch
- torchvision
- pyyaml

Install
-------
Install Python packages (preferably in a virtualenv):

```bash
pip install -r requirements.txt
```

Ensure the `compressai/` folder is present at the project root (the project expects to import `compressai.zoo`).

Run training
------------

```bash
python scripts/train.py
```

This will run the trainer for 2 iterations on random data and print loss, distortion and rate.

Project structure
-----------------
See top-level structure in the repository root. Key modules live under `src/`.



Cài cuda12.6
conda activate vfc
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
nano $CONDA_PREFIX/etc/conda/activate.d/cuda.sh

export CUDA_HOME=/work/u9564043/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nano $CONDA_PREFIX/etc/conda/deactivate.d/cuda.sh
unset CUDA_HOME

sudo apt-get install cmake g++
cd src
mkdir build
cd build
conda activate $YOUR_PY36_ENV_NAME
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j