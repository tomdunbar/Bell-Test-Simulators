# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:40:48 2026

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
# Traditional QM singlet outcomes Monte Carlo
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

    # Compute mean safely
    E = np.full(num_bins, np.nan)
    nonzero = counts > 0
    E[nonzero] = sums[nonzero] / counts[nonzero]

    return E


def PXYgamma(Xoutcome, Youtcome, X, Y, gamma, bins):
    """
    Compute P(X = Xoutcome and Y = Youtcome) binned by gamma.

    Parameters
    ----------
    Xoutcome, Youtcome : desired outcomes, usually ±1
    X, Y : arrays of ±1 outcomes
    gamma : array of angles or other binning variable
    bins : array of bin edges

    Returns
    -------
    P : array of probabilities per bin
    """

    # Bin index for each gamma value
    bin_indices = np.digitize(gamma, bins) - 1

    num_bins = len(bins) - 1

    # Remove out-of-range values
    valid = (bin_indices >= 0) & (bin_indices < num_bins)

    bin_indices = bin_indices[valid]
    X = X[valid]
    Y = Y[valid]

    # Count total samples per bin
    counts = np.bincount(bin_indices, minlength=num_bins)

    # Indicator for the desired joint outcome
    matches = (X == Xoutcome) & (Y == Youtcome)

    # Count matching samples per bin
    match_counts = np.bincount(
        bin_indices,
        weights=matches.astype(int),
        minlength=num_bins
    )

    # Compute probability safely
    P = np.full(num_bins, np.nan)
    nonzero = counts > 0
    P[nonzero] = match_counts[nonzero] / counts[nonzero]  #This is the 'frequentist' def of probability

    return P

# ------------------------------------
# Random measurement angles
# ------------------------------------
a = np.random.uniform(0, 2*np.pi, N)
b = np.random.uniform(0, 2*np.pi, N)
# Compute delta

delta_ab = (a - b) % (2*np.pi)


# ------------------------------------
# Experimental Data
# This is generated based on traditional QM theory
# We are using it as our experimental data to test new thoughts against.
# ------------------------------------
X,Y = QMtheoryPair(a, b)

E_exp = EXYdelta(X,Y,delta_ab , bins)

Ppp_exp = PXYgamma(1, 1, X, Y,delta_ab, bins)
#Ppn_exp = PXYgamma(1, -1, X, Y,delta_ab, bins)
#Pnp_exp = PXYgamma(-1, 1, X, Y,delta_ab, bins)
#Pnn_exp = PXYgamma(-1, -1, X, Y,delta_ab, bins)

# ------------------------------------
# New Theory ideas
# ------------------------------------
# Local hidden variable angle
lam = np.random.uniform(0, 2*np.pi, N)

# Local deterministic responses
A = np.sign(np.cos(lam + a))
B = np.sign(np.cos(lam + np.pi  + b))

E_new = EXYdelta(A,B,delta_ab , bins)

Ppp_new = PXYgamma(1, 1, A, B,delta_ab, bins)
#Ppn_new = PXYgamma(1, -1, A, B,delta_ab, bins)
#Pnp_new = PXYgamma(-1, 1, A, B,delta_ab, bins)
#Pnn_new = PXYgamma(-1, -1, A, B,delta_ab, bins)


plt.figure(figsize=(8,5))

#Traiditional Bell Test Expectation value plot
# Experimental Data
plt.scatter(bin_centers, E_exp,
            color="red",
            s=25,
            alpha=0.8,
            label="Experimental Data")

plt.scatter(bin_centers, E_new,
            color="blue",
            s=25,
            alpha=0.8,
            label="Simulated Data from LHV")

#check of my functions
# plt.scatter(bin_centers, Ppp - Ppn - Pnp + Pnn,
#             color="tab:green",
#             s=25,
#             alpha=0.8,
#             label="Experimental Data")

plt.xlabel(r"$a-b$, difference of Alice and Bob's angle")
plt.ylabel(r"$E(a-b)$")
plt.title("Traditional Expectation Value Data")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


### New ways of analyzing data
plt.figure(figsize=(8,5))


# Experimental Data
plt.scatter(bin_centers, Ppp_exp,
            color="tab:red",
            s=25,
            alpha=0.8,
            label="P++")

# plt.scatter(bin_centers, Ppn_exp,
#             color="red",
#             s=25,
#             alpha=0.8,
#             label="P+-")


# plt.scatter(bin_centers, Pnp_exp,
#             color="green",
#             s=25,
#             alpha=0.8,
#             label="P-+")


# plt.scatter(bin_centers, Pnn_exp,
#             color="magenta",
#             s=25,
#             alpha=0.8,
#             label="P--")

#New thoughts
plt.scatter(bin_centers, Ppp_new,
            color="tab:blue",
            s=25,
            alpha=0.8,
            label="P++ New")



plt.xlabel(r"$a-b$, difference of Alice and Bob's angle")
plt.ylabel(r"$P$")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

