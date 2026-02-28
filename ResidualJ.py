# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 09:28:27 2026

@author: Tom Dunbar
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Parameters
# ------------------------------------
N = 500_000       # number of trials

num_bins = 100       # bins for delta histogram
bins = np.linspace(0, 2*np.pi, num_bins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

def QMtheoryPair(x,y):
# ------------------------------------
# Quantum singlet outcomes Monte Carlo
# ------------------------------------
    #collaspe the wavefcn by measuring on x    
    X = np.random.choice([-1, 1], size=N)
    
    #Probability that X == Y, given x and y
    p_equal = (1 - np.cos(x-y)) / 2
    
    rn = np.random.uniform(0, 1, N)  #random number
    # If rn < p_equal → Y = X
    # else → Y = -X
    Y = np.where(rn < p_equal, X, -X)
    
    return X,Y


def EXYdelta(X, Y, delta, bins):
    """
    Compute correlation E(XY) binned by delta (fully vectorized).

    Parameters
    ----------
    X, Y : arrays of ±1 outcomes
    delta : array of angle differences (same length)
    bins : array of bin edges

    Returns
    -------
    E : array of correlations per bin
    """

    # Bin index for each delta (0 to num_bins-1)
    bin_indices = np.digitize(delta, bins) - 1

    num_bins = len(bins) - 1

    # Remove out-of-range values (just in case)
    valid = (bin_indices >= 0) & (bin_indices < num_bins)

    bin_indices = bin_indices[valid]
    XY = (X * Y)[valid]

    # Count how many samples per bin
    counts = np.bincount(bin_indices, minlength=num_bins)

    # Sum of XY per bin
    sums = np.bincount(bin_indices, weights=XY, minlength=num_bins)

    # Compute mean
    E = np.full(num_bins, np.nan)
    nonzero = counts > 0
    E[nonzero] = sums[nonzero] / counts[nonzero]

    return E

def Detector_J(x,LHV_angle,J_res):
    
    X = np.sign(np.cos(x - np.angle(np.exp(1j*LHV_angle) + J_res)))
    
    J_res = np.exp(1j*LHV_angle) - np.exp(1j *X*x)  #initial J - final J
    
    
    return X,J_res

# ------------------------------------
# Random measurement angles
# ------------------------------------
a = np.random.uniform(0, 2*np.pi, N)
b = np.random.uniform(0, 2*np.pi, N)
# Compute delta

delta_ab = (a - b) % (2*np.pi)


# ------------------------------------
# Residual J LHV Monte Carlo
# ------------------------------------
# Local hidden variable angle
lam = np.random.uniform(0, 2*np.pi, N)

Ja_res = 0
Jb_res = 0

A = np.zeros(N)
B = np.zeros(N)

for iteration in range(0,N):
    a_angle = a[iteration]
    b_angle = b[iteration]
    lam_angle = lam[iteration] 

    # Local deterministic responses
    A[iteration],Ja_res = Detector_J(a_angle,lam_angle,Ja_res)
    B[iteration],Jb_res = Detector_J(b_angle,lam_angle+np.pi,Jb_res)
    
    

E_lhv = EXYdelta(A,B,delta_ab, bins)


# ------------------------------------
# Ideal and Monte Carlo results
# ------------------------------------
delta_fine = np.linspace(0, 2*np.pi, 500)
E_qm_ideal  = -np.cos(delta_fine)
E_lhv_ideal = -2/np.pi * np.arcsin(np.cos(delta_fine))  # triangle
E_lmc_ideal = -2/np.pi * np.cos(delta_fine)  #local max correlation

plt.figure(figsize=(8,5))

# Ideal curves
plt.plot(delta_fine, E_qm_ideal,
         color="tab:blue",
         linewidth=2.5,
         label="Ideal QM  $-\\cos(a-b)$")

plt.plot(delta_fine, E_lhv_ideal,
         color="tab:red",
         linewidth=2.5,
         label="Typical LHV model")

plt.plot(delta_fine, E_lmc_ideal,
         color="tab:green",
         linewidth=2.5,
         label="LMC model")


plt.scatter(bin_centers, E_lhv,
            color="firebrick",
            s=25,
            alpha=0.8,
            label="Residual J LHV Monte Carlo")


# plt.scatter(bin_centers, E_sc,
#             color="magenta",
#             s=25,
#             alpha=0.8,
#             label="Scaled Cosine Correlation Monte Carlo")

plt.xlabel(r"$a-b$, difference of Alice and Bob's angle")
plt.ylabel(r"$E(a-b)$")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()