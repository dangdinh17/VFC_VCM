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
