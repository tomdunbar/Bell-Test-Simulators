# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:49:00 2026

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

def compute_probabilities(data):

    # check for invalid values
    invalid_values = [x for x in data if x not in (1, -1)]
    if invalid_values:
        print(f"Error: unexpected values found: {set(invalid_values)}")
        return

    total = len(data)
    count_pos = sum(1 for x in data if x == 1)
    count_neg = sum(1 for x in data if x == -1)

    p_pos = count_pos / total
    p_neg = count_neg / total

    print(f"p(1)  = {p_pos:.2f}")
    print(f"p(-1) = {p_neg:.2f}")
    
    
#Reference Frame Response function
def RefFrameResp(x,y):
    #Probability that X == Y, given x and y
    p_equal = np.abs(np.cos(x-y)/2) +1/2
    
    
    rn = np.random.uniform(0, 1, len(x))  #random numbers

    # If rn < p_equal → +1 (up)
    # else → -1 (down)
    X = np.where(rn < p_equal, 1, -1)
    return X


# ------------------------------------
# Random Reference Frame Angles
# ------------------------------------
lamA = np.random.uniform(0, 2*np.pi, N)
lamB = np.random.uniform(0, 2*np.pi, N)

# ------------------------------------
# Random measurement angles
# ------------------------------------
alpha = np.random.uniform(0, 2*np.pi, N)
beta = np.random.uniform(0, 2*np.pi, N)
# Compute delta

delta_ab = (alpha - beta) % (2*np.pi)

# ------------------------------------
# Signed-measure LHV Monte Carlo
# ------------------------------------
# Local deterministic responses
#A = np.sign(np.cos(alpha - lamA))
#B = np.sign(np.cos(beta - lamA + np.pi))


A = RefFrameResp(alpha, lamA)
B = RefFrameResp(beta, lamB)

compute_probabilities(A)
compute_probabilities(B)

E_lhv = EXYdelta(A, B, delta_ab, bins)

# ------------------------------------
# Ideal and Monte Carlo results
# ------------------------------------
delta_fine = np.linspace(0, 2*np.pi, 500)
E_qm_ideal  = -np.cos(delta_fine)
E_lhv_ideal = -2/np.pi * np.arcsin(np.cos(delta_fine))  # triangle


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



plt.scatter(bin_centers, E_lhv,
            color="firebrick",
            s=25,
            alpha=0.8,
            label="Reference Frame LHV Monte Carlo")


plt.xlabel(r"$a-b$, difference of Alice and Bob's angle")
plt.ylabel(r"$E(a-b)$")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()