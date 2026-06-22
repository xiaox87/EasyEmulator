"""
GP emulation for power spectrum P(k).
Demonstrates optional normalization of inputs and outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from util.process_data import norm_normal, reverse_norm_normal
import os

os.makedirs('./fig', exist_ok=True)

# Load data
params = np.load('./data/ps_cosmo.npy')
ps = np.load('./data/ps.npy')

# Split into train and test (20% test)
X_train, X_test, y_train, y_test = train_test_split(params, ps, test_size=0.2)

# ---- Input preprocessing ----
use_x_norm = True   # set False to skip normalization of inputs

if use_x_norm:
    X_train_proc, x_state = norm_normal(X_train)
    X_test_proc = (X_test - x_state['mean']) / x_state['std']
else:
    X_train_proc = X_train
    X_test_proc = X_test

# ---- Output preprocessing ----
use_y_norm = True   # set False to skip normalization of outputs

if use_y_norm:
    y_train_proc, y_state = norm_normal(y_train)
    y_test_proc = (y_test - y_state['mean']) / y_state['std']
else:
    y_train_proc = y_train
    y_test_proc = y_test
    y_state = None

# ---- GP training ----
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gp.fit(X_train_proc, y_train_proc)
print("GP trained.")

# ---- Prediction ----
y_pred_proc = gp.predict(X_test_proc)

# Reverse output preprocessing if needed
if use_y_norm:
    y_pred = reverse_norm_normal(y_pred_proc, y_state)
    y_true = y_test
else:
    y_pred = y_pred_proc
    y_true = y_test_proc

# k array
k = np.logspace(np.log10(0.001), np.log10(0.1), y_true.shape[1])

# ---- Visualisation ----
n_show = min(3, len(y_true))
colors = ['red', 'blue', 'green']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for i in range(n_show):
    ax1.plot(k, y_true[i], color=colors[i], label=f'True {i+1}')
    ax1.plot(k, y_pred[i], '--', color=colors[i], label=f'Pred {i+1}')
ax1.set_xlabel('k (h/Mpc)')
ax1.set_ylabel('P(k) (Mpc/h)^3')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.set_title('Power spectra (true vs predicted)')

for i in range(n_show):
    rel_err = np.abs(y_pred[i] - y_true[i]) / (np.abs(y_true[i]) + 1e-8)
    ax2.plot(k, rel_err, color=colors[i], label=f'Error {i+1}')
ax2.set_xlabel('k (h/Mpc)')
ax2.set_ylabel('Relative error')
ax2.set_xscale('log')
ax2.legend()
ax2.set_title('Relative errors')

plt.tight_layout()
plt.savefig('./fig/ps_emulation.png', dpi=150)
print("Figure saved to ./fig/ps_emulation.png")
plt.show()

# Overall mean relative error
mean_rel_err_all = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8))
print(f"Overall mean relative error (all test): {mean_rel_err_all:.3f}")