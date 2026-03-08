# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:31:28 2026

@author: Gamer
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Parameters
# ------------------------------------
N = 1_000_000       # number of trials
num_bins = 100       # bins for delta histogram
bins = np.linspace(0, 2*np.pi, num_bins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

def QMtheoryPair(x,y):
# ------------------------------------
# Quantum singlet outcomes Monte Carlo
# ------------------------------------
    #collaspe the wavefcn by measuring on x    
    X = np.random.choice([-1, 1], size=len(x))
    
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

    # Compute mean safely
    E = np.full(num_bins, np.nan)
    nonzero = counts > 0
    E[nonzero] = sums[nonzero] / counts[nonzero]

    return E

# ------------------------------------
# Random measurement angles
# ------------------------------------
a = np.random.uniform(0, 2*np.pi, N)
b = np.random.uniform(0, 2*np.pi, N)
# Compute delta

delta_ab = (a - b) % (2*np.pi)

# ------------------------------------
#  LHV Monte Carlo with Non-Diophantine arithmetic
# ------------------------------------


def f(x):
    """
    Non-Diophantine generator (Eq. 57–59 style).
    Vectorized and numerically safe.
    """

    x = np.asarray(x)

    n = np.floor(x / 2)

    # argument inside sqrt
    inner = 2*x - n

    # enforce valid domain
    inner = np.clip(inner, 0.0, 1.0)

    arg = np.sqrt(inner)
    arg = np.clip(arg, -1.0, 1.0)

    xprime = n/2 + (1/np.pi)*np.arcsin(arg)

    return xprime


def finv(y):
    """
    Inverse generator.
    """

    y = np.asarray(y)

    n = np.floor(2*y)

    theta = np.pi*(y - n/2)

    # recover original branch expression
    inner = np.sin(theta)**2

    x = (n + inner) / 2

    return x


# Local deterministic responses
# Local hidden variable angle
lam = np.random.uniform(0, 2*np.pi, N)

lamPrime = f(lam)

A = np.sign(np.cos(lam - a))
B = -np.sign(np.cos(lam - b))

E_lhv = EXYdelta(A,B,delta_ab, bins)
# ------------------------------------
# Quantum singlet outcomes Monte Carlo
# ------------------------------------
X,Y = QMtheoryPair(a, b)

E_QM = EXYdelta(X,Y,delta_ab , bins)

# ------------------------------------
# Ideal and Monte Carlo results
# ------------------------------------
delta_fine = np.linspace(0, 2*np.pi, 500)

E_pred = -2/np.pi * np.arcsin(np.cos(delta_fine))  # triangle

E_exp  = -np.cos(delta_fine)


# plot
plt.figure(figsize=(8,6))

plt.plot(delta_fine, E_pred, label="Predicted (local hidden variable)", linewidth=3)
plt.plot(delta_fine, E_exp, label="Experiment", linewidth=3)

plt.xlabel("a-b")
plt.ylabel("Correlation ⟨AB⟩")
plt.title("Bell Test: Prediction vs Experiment")

plt.legend()
plt.grid(True)

plt.show()