# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 07:16:27 2026

@author: Tom Dunbar
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
# Quantum singlet, but correlated,  Monte Carlo
# ------------------------------------
    #collaspe the wavefcn by measuring on x    
    X = np.random.choice([-1, 1], size=N)
    
    #Probability that X == Y, given x and y
    p_equal = (1 + np.cos(x+y)) / 2
    
    rn = np.random.uniform(0, 1, N)  #random number
    P = np.where(rn < p_equal, 1, -1)
    
    # Step 3: determine Z
    Y = P * X
    
    return X,Y


def ghz3_sample(x, y, z):   #alternate understanding
    """
    Sample (X, Y, Z) in {+1, -1} for equatorial measurements
    on a 3-qubit GHZ state with measurement angles x, y, z.
    
    Parameters:
        x, y, z : float
            Measurement angles (radians)
    
    Returns:
        (X, Y, Z) : tuple of ints (+1 or -1)
    """
    # Step 1: sample X and Y uniformly
    X = np.random.choice([-1, 1], size=N)
    Y = np.random.choice([-1, 1], size=N)
    
    # Step 2: sample product P = XYZ
    p_equal = (1 + np.cos(x + y + z)) / 2
    rn = np.random.uniform(0, 1, N)  #random number
    
    P = np.where(rn < p_equal, 1, -1)
    
    # Step 3: determine Z
    Z = P * X * Y
    
    return X, Y, Z


def E_delta(*arrays, delta, bins):
    """
    Compute correlation E(product of arrays) binned by delta.

    Parameters
    ----------
    *arrays : any number of ±1 arrays (same length)
    delta   : array (same length)
    bins    : array of bin edges

    Returns
    -------
    E : array of correlations per bin
    """

    if len(arrays) == 0:
        raise ValueError("At least one outcome array must be provided.")

    # Stack arrays into shape (num_arrays, N)
    stacked = np.stack(arrays)

    # Product along first axis → shape (N,)
    product = np.prod(stacked, axis=0)

    # Bin index for each delta (0 to num_bins-1)
    bin_indices = np.digitize(delta, bins) - 1
    num_bins = len(bins) - 1

    # Keep valid bin entries
    valid = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = bin_indices[valid]
    product = product[valid]

    # Counts per bin
    counts = np.bincount(bin_indices, minlength=num_bins)

    # Weighted sum per bin
    sums = np.bincount(bin_indices, weights=product, minlength=num_bins)

    # Compute means safely
    E = np.full(num_bins, np.nan)
    nonzero = counts > 0
    E[nonzero] = sums[nonzero] / counts[nonzero]

    return E

# ------------------------------------
# Random measurement angles
# ------------------------------------
a = np.random.uniform(0, 2*np.pi, N)
b = np.random.uniform(0, 2*np.pi, N)
c = np.random.uniform(0, 2*np.pi, N)

# Compute deltas
delta_ab = (a + b) % (2*np.pi)
delta_ac = (a + c) % (2*np.pi)
delta_bc = (b + c) % (2*np.pi)


# -----------------------------------
# GHZ sampling (vectorized)
# -----------------------------------
#A,B,C = QMtheoryGHZ(a, b, c)
A,B,C =  ghz3_sample(a, b, c)


# Compute pairwise correlations
E_AB = E_delta(A, B, delta = delta_ab, bins = bins)
E_AC = E_delta(A, C, delta = delta_ac, bins = bins)
E_BC = E_delta(B, C, delta = delta_bc, bins = bins)

delta_abc = (a+b+c) % (2*np.pi)

E_ABC = E_delta(A, B, C, delta = delta_abc, bins = bins)

# -----------------------------------
# Three pairs
# -----------------------------------
A1,B1 = QMtheoryPair(a, b)
A2,C2 = QMtheoryPair(a, c)
B3,C3 = QMtheoryPair(b, c)


E_A1B1 = E_delta(A1, B1, delta = delta_ab, bins = bins)
E_A2C2 = E_delta(A2, C2, delta = delta_ac, bins = bins)
E_B3C3 = E_delta(B3, C3, delta = delta_bc, bins = bins)

E_B1C2 = E_delta(B1, C2, delta = delta_bc, bins = bins)



# ------------------------------------
# Ideal and Monte Carlo results
# ------------------------------------
delta_fine = np.linspace(0, 2*np.pi, 500)
E_qm_ideal  = np.cos(delta_fine)
#E_BC_ideal = np.cos(delta_fine) * np.cos(delta_fine)
E_BC_ideal = 1/2*np.cos(delta_fine)

plt.figure(figsize=(8,5))

# Ideal curves
plt.plot(delta_fine, E_qm_ideal,
         color="tab:blue",
         linewidth=1.5,
         label="$\\cos(\Delta)$")

# plt.plot(delta_fine, E_BC_ideal,
#          color="tab:green",
#          linewidth=1.5,
#          #label="Ideal QM  $\\cos(a-b)\\cos(a-c)$"
#          label="$0.5\\cos(\Delta)$")

# Monte Carlo points
plt.scatter(bin_centers, E_AB,
            color="royalblue",
            s=25,
            alpha=0.8,
            label="E(A,B)")

plt.scatter(bin_centers, E_AC,
            color="red",
            s=25,
            alpha=0.8,
            label="E(A,C)")

plt.scatter(bin_centers, E_BC,
            color="green",
            s=25,
            alpha=0.8,
            label="E(B,C)")

plt.scatter(bin_centers, E_ABC,
            color="magenta",
            s=25,
            alpha=0.8,
            label="E(A,B,C)")

plt.xlabel(r"$\Delta$, sum of angles")
plt.ylabel(r"$E$")
plt.title("One GHZ Tripple")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,5))

# Ideal curves
plt.plot(delta_fine, E_qm_ideal,
         color="tab:blue",
         linewidth=1.5,
         label="$\\cos(\Delta)$")

# Monte Carlo points
plt.scatter(bin_centers, E_A1B1,
            color="royalblue",
            s=25,
            alpha=0.8,
            label="E(A1,B1)")

plt.scatter(bin_centers, E_A2C2,
            color="red",
            s=25,
            alpha=0.8,
            label="E(A2,C2)")

plt.scatter(bin_centers, E_B3C3,
            color="green",
            s=25,
            alpha=0.8,
            label="E(B3,C3)")

plt.scatter(bin_centers, E_B1C2,
            color="magenta",
            s=25,
            alpha=0.8,
            label="E(B1,C2)")

plt.xlabel(r"$\Delta$, sum in angle")
plt.ylabel(r"$E$")
plt.title("Three Entangled Pairs")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
