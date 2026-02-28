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

def QMtheoryGHZ(x,y,z):  #my understanding of how the GHZ state is
# ------------------------------------
# Quantum GHZ state
# ------------------------------------
    #collaspe the wavefcn by measuring on x    
    X = np.random.choice([-1, 1], size=N)
    
    #Probability that X == Y, given x and y
    p_equal = (1 - np.cos(x-y)) / 2
    
    rn = np.random.uniform(0, 1, N)  #random number
    # If rn < p_equal → Y = X
    # else → Y = -X
    Y = np.where(rn < p_equal, X, -X)
    
    #Probability that X == Z, given x and Z
    p_equal = (1 - np.cos(x-z)) / 2
    
    rn = np.random.uniform(0, 1, N)  #random number
    # If rn < p_equal → Z = X
    # else → Z = -X
    Z = np.where(rn < p_equal, X, -X)
    
    return X,Y,Z


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


def EXYZdelta(X, Y, Z,delta, bins):
    """
    Compute correlation E(XY) binned by delta (fully vectorized).

    Parameters
    ----------
    X, Y Z: arrays of ±1 outcomes
    delta : asum of angles (same length)
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
    XYZ = (X * Y * Z)[valid]

    # Count how many samples per bin
    counts = np.bincount(bin_indices, minlength=num_bins)

    # Sum of XY per bin
    sums = np.bincount(bin_indices, weights=XYZ, minlength=num_bins)

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
c = np.random.uniform(0, 2*np.pi, N)

# Compute deltas
delta_ab = (a - b) % (2*np.pi)
delta_ac = (a - c) % (2*np.pi)
delta_bc = (b - c) % (2*np.pi)


# -----------------------------------
# GHZ sampling (vectorized)
# -----------------------------------
#A,B,C = QMtheoryGHZ(a, b, c)
A,B,C =  ghz3_sample(a, b, c)


# Compute pairwise correlations
E_AB = EXYdelta(A, B, delta_ab, bins)
E_AC = EXYdelta(A, C, delta_ac, bins)
E_BC = EXYdelta(B, C, delta_bc, bins)

E_ABC = EXYZdelta(A, B, C, a+b+c, bins)

# -----------------------------------
# Three pairs
# -----------------------------------
A1,B1 = QMtheoryPair(a, b)
A2,C2 = QMtheoryPair(a, c)
B3,C3 = QMtheoryPair(b, c)


E_A1B1 = EXYdelta(A1, B1, delta_ab, bins)
E_A2C2 = EXYdelta(A2, C2, delta_ac, bins)
E_B3C3 = EXYdelta(B3, C3, delta_bc, bins)

E_B1C2 = EXYdelta(B1, C2, delta_bc, bins)



# ------------------------------------
# Ideal and Monte Carlo results
# ------------------------------------
delta_fine = np.linspace(0, 2*np.pi, 500)
E_qm_ideal  = -np.cos(delta_fine)
#E_BC_ideal = np.cos(delta_fine) * np.cos(delta_fine)
E_BC_ideal = 1/2*np.cos(delta_fine)

plt.figure(figsize=(8,5))

# Ideal curves
plt.plot(delta_fine, E_qm_ideal,
         color="tab:blue",
         linewidth=1.5,
         label="$-\\cos(\Delta)$")

plt.plot(delta_fine, E_BC_ideal,
         color="tab:green",
         linewidth=1.5,
         #label="Ideal QM  $\\cos(a-b)\\cos(a-c)$"
         label="$0.5\\cos(\Delta)$")

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

plt.xlabel(r"$\Delta$, difference in angle")
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
         label="$-\\cos(\Delta)$")

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

plt.xlabel(r"$\Delta$, difference in angle")
plt.ylabel(r"$E$")
plt.title("Three Entangled Pairs")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
