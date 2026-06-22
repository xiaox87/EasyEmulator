"""
Generate synthetic cosmological data: power spectrum and 2-point correlation function.
Uses CAMB for linear power spectrum and computes xi(r) via numerical integration.
"""

import numpy as np
import camb
from scipy.integrate import simps


def generate_cosmology_params(Ncosmo=100, params_config=None):
    """
    Generate random cosmological parameters uniformly within ±5σ around Planck.

    Args:
        Ncosmo: number of samples
        params_config: dict with keys 'central' and 'sigma' (optional)

    Returns:
        params: array of shape (Ncosmo, n_params)
        param_names: list of parameter names
    """
    if params_config is not None:
        central, sigma = params_config['central'], params_config['sigma']
    else:
        central = {
            'Om': 0.315, 'Ob': 0.049, 'As': 2.1e-9, 'ns': 0.965,
            'w0': -1.0, 'h': 0.674, 'mnu': 0.06
        }
        sigma = {
            'Om': 0.007, 'Ob': 0.0006, 'As': 0.15e-9, 'ns': 0.004,
            'w0': 0.1, 'h': 0.005, 'mnu': 0.01
        }

    names = list(central.keys())
    nparams = len(names)
    params = np.zeros((Ncosmo, nparams))
    for i, name in enumerate(names):
        low = central[name] - 5 * sigma[name]
        high = central[name] + 5 * sigma[name]
        params[:, i] = np.random.uniform(low, high, Ncosmo)
    return params, names


def compute_power_spectrum(params, kmin=0.001, kmax=0.1, nk=200):
    """
    Compute linear matter power spectrum P(k) for given cosmological parameters.

    Args:
        params: array of length 7 in order: Om, Ob, As, ns, w0, h, mnu
        kmin, kmax: min and max k in h/Mpc
        nk: number of k points (log-spaced)

    Returns:
        k: array of k values (h/Mpc)
        P: array of P(k) in (Mpc/h)^3
    """
    Om, Ob, As, ns, w0, h, mnu = params[:7]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100*h, ombh2=Ob*h**2, omch2=(Om-Ob)*h**2, mnu=mnu)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_dark_energy(w=w0)
    pars.set_matter_power(redshifts=[0.], kmax=1.0)
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=nk)
    return kh.flatten(), pk[0, :]


def compute_xi_from_pk(k, P, r_array):
    """
    Compute correlation function xi(r) from P(k) via:
        xi(r) = 1/(2π²) ∫ P(k) k² sin(kr)/(kr) dk
    Uses logarithmic integration (k is log-spaced).
    """
    xi = np.zeros_like(r_array)
    for i, r in enumerate(r_array):
        if r == 0:
            continue
        kr = k * r
        integrand = P * k**2 * np.sin(kr) / (kr)
        xi[i] = simps(integrand * k, np.log(k)) / (2 * np.pi**2)
    return xi


def compute_2pcf(params, rmin=15, rmax=120, nr=50, kmin=0.001, kmax=0.1, nk=200):
    """
    Compute two-point correlation function xi(r) for given parameters.
    Returns r array and xi(r).
    """
    k, P = compute_power_spectrum(params, kmin, kmax, nk)
    r = np.logspace(np.log10(rmin), np.log10(rmax), nr)
    xi = compute_xi_from_pk(k, P, r)
    return r, xi


def add_noise(data, err_percent=0.03):
    """
    Add multiplicative Gaussian noise to data.
    Each element is multiplied by a random factor ~ Normal(1, err_percent).
    """
    noise = np.random.normal(1.0, err_percent, size=data.shape)
    return data * noise


def generate_data(Ncosmo=100, err_percent=0.03, save_dir='./data'):
    """
    Generate cosmology samples, power spectra and correlation functions.
    Saves:
        ps_cosmo.npy: cosmological parameters (Ncosmo, nparams)
        ps.npy: power spectra (Ncosmo, nk)
        tpcf_cosmo.npy: cosmological parameters (same as ps_cosmo)
        tpcf.npy: correlation functions (Ncosmo, nr)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Generate parameters
    params, names = generate_cosmology_params(Ncosmo)
    np.save(os.path.join(save_dir, 'ps_cosmo.npy'), params)
    np.save(os.path.join(save_dir, 'tpcf_cosmo.npy'), params)

    # Setup k and r grids
    nk = 200
    kmin, kmax = 0.001, 0.1
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    nr = 50
    rmin, rmax = 15, 120
    r = np.logspace(np.log10(rmin), np.log10(rmax), nr)

    ps = np.zeros((Ncosmo, nk))
    tpcf = np.zeros((Ncosmo, nr))

    # Single loop: compute P(k) once per cosmology, then xi(r)
    for i in range(Ncosmo):
        if i % 10 == 0:
            print(f"Generating sample {i} (ps and tpcf)...")
        k_i, P_i = compute_power_spectrum(params[i], kmin, kmax, nk)
        ps[i] = P_i
        xi_i = compute_xi_from_pk(k_i, P_i, r)
        tpcf[i] = xi_i

    # Add noise to both datasets
    ps_noisy = add_noise(ps, err_percent)
    tpcf_noisy = add_noise(tpcf, err_percent)

    np.save(os.path.join(save_dir, 'ps.npy'), ps_noisy)
    np.save(os.path.join(save_dir, 'tpcf.npy'), tpcf_noisy)

    print(f"Data generated and saved to {save_dir}")
    print(f"  Parameters shape: {params.shape}")
    print(f"  Power spectra shape: {ps_noisy.shape}")
    print(f"  Correlation functions shape: {tpcf_noisy.shape}")