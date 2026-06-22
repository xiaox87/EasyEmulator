"""
Check generated data: plot power spectra and correlation functions for all samples,
and compare to the fiducial (central) cosmology.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from util.data import compute_power_spectrum, compute_2pcf

os.makedirs('./fig', exist_ok=True)

# Load generated data
ps_cosmo = np.load('./data/ps_cosmo.npy')
ps = np.load('./data/ps.npy')
tpcf_cosmo = np.load('./data/tpcf_cosmo.npy')
tpcf = np.load('./data/tpcf.npy')

# Fiducial cosmology (central values) in order: Om, Ob, As, ns, w0, h, m_nv
fiducial_params = np.array([
    0.315, 0.049, 2.1e-9, 0.965, -1.0, 0.674, 0.06
])

# Compute fiducial power spectrum and correlation function
k_fid, ps_fid = compute_power_spectrum(fiducial_params)
r_fid, xi_fid = compute_2pcf(fiducial_params)

# k and r arrays (same for all samples)
k = np.logspace(np.log10(0.001), np.log10(0.1), ps.shape[1])
r = np.logspace(np.log10(15), np.log10(120), tpcf.shape[1])

# ---- Power spectra ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].set_title('Power spectra P(k)')
for i in range(ps.shape[0]):
    axes[0].plot(k, ps[i], color='blue', alpha=0.1, lw=0.5)
axes[0].plot(k, ps_fid, color='black', lw=2, label='Fiducial')
axes[0].set_xlabel('k (h/Mpc)')
axes[0].set_ylabel('P(k) (Mpc/h)^3')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].legend()

axes[1].set_title('Relative difference from fiducial')
for i in range(ps.shape[0]):
    rel_diff = (ps[i] - ps_fid) / ps_fid
    axes[1].plot(k, rel_diff, color='red', alpha=0.1, lw=0.5)
axes[1].axhline(0, color='black', linestyle='--', lw=1)
axes[1].set_xlabel('k (h/Mpc)')
axes[1].set_ylabel('(P - P_fid) / P_fid')
axes[1].set_xscale('log')

plt.tight_layout()
plt.savefig('./fig/ps_check.png', dpi=150)
print("Power spectrum check figure saved to ./fig/ps_check.png")
plt.close()

# ---- Correlation functions ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].set_title('Correlation functions xi(r)')
for i in range(tpcf.shape[0]):
    axes[0].plot(r, tpcf[i], color='blue', alpha=0.1, lw=0.5)
axes[0].plot(r, xi_fid, color='black', lw=2, label='Fiducial')
axes[0].set_xlabel('r (Mpc/h)')
axes[0].set_ylabel('xi(r)')
axes[0].set_xscale('log')
axes[0].legend()

axes[1].set_title('Relative difference from fiducial')
for i in range(tpcf.shape[0]):
    mask = np.abs(xi_fid) > 1e-10
    rel_diff = np.full_like(xi_fid, np.nan)
    rel_diff[mask] = (tpcf[i][mask] - xi_fid[mask]) / xi_fid[mask]
    axes[1].plot(r, rel_diff, color='red', alpha=0.1, lw=0.5)
axes[1].axhline(0, color='black', linestyle='--', lw=1)
axes[1].set_xlabel('r (Mpc/h)')
axes[1].set_ylabel('(xi - xi_fid) / xi_fid')
axes[1].set_xscale('log')
axes[1].set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('./fig/tpcf_check.png', dpi=150)
print("Correlation function check figure saved to ./fig/tpcf_check.png")
plt.close()

print("Data check completed.")