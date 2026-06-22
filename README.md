# Simple GP Emulator for Cosmological Observables

This repository demonstrates building a Gaussian Process (GP) emulator for two key cosmological observables:
- **Power spectrum**
- **Two-point correlation function**

The code generates synthetic data using the [CAMB](https://camb.info/) Boltzmann code, then trains a GP to map cosmological parameters to these observables. It includes options for data normalization and visualisation of results.

## Features
- Generate synthetic training data with randomised cosmological parameters (within ±5σ of Planck).
- Compute linear matter power spectrum and corresponding correlation function.
- Train a GP emulator using `scikit-learn`.
- Optional normalisation of inputs and outputs to improve GP performance.
- Visual comparison of true vs predicted spectra and relative errors.

## Requirements

### Data Generation (optional)
- `camb` – for power spectrum computation
- `scipy` – for numerical integration

### Running the Emulator
- `numpy`
- `matplotlib`
- `scikit-learn`

Install all required packages with:
```bash
pip install numpy matplotlib scikit-learn camb scipy
