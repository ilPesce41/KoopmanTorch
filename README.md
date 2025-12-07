# Introduction
[![DOI](https://zenodo.org/badge/893574206.svg)](https://doi.org/10.5281/zenodo.17850359)

This repository contains source code for training an Autoencoder to approximate Koopman Operators for dynamical systems.

Training routines are implemented with Pytorch Lightning.

This architecture was first published by Lusch et al 2018 [1]

Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton. "Deep learning for universal linear embeddings of nonlinear dynamics." Nature communications 9.1 (2018): 4950.

# Installation

```
git clone git@github.com:ilPesce41/KoopmanTorch.git
cd KoopmanTorch
pip install .
```


# Examples

This repository contains examples with basic dynamical systems


## Simple Dynamical System

```.sh
python generate_discrete_dataset.py
python train_discrete.py
python test_discrete.py
```

## Simple Pendulum

```.sh
python generate_pendulum_dataset.py
python train_pendulum.py
python test_pendulum.py
```