# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:03:30 2026

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

def ang2campter(theta):
    return 1/(np.sin(theta/ (1/2)))**2

def campter2ang(C):
    return 1/2 * (np.sin(C))**2


def f(x):
    """
    Outer to inner
    
    Original fcn
    # x = np.asarray(x)
    # x = x % (2*np.pi)     #restrict to 0 to 2pi
    # x = x/(2*np.pi)   #normalize to 0 to 1
    # n = np.floor(2*x)
    # result =  n/2 + (1/np.pi)*np.arcsin(np.sqrt(2*x - n))
    # return  (2*np.pi)*result  #return to 0 to 2pi
    
    Updated, for cleaner implimentation
    Instead of computing floor, let modf split: 2x=n+r
    where:
        n = integer part
        r = fractional part
    and r = 2x - n, which is exactly the quantity inside the square root.
    """

    x = np.asarray(x) % (2*np.pi)  #restrict to 0 to 2pi

    u = x/(2*np.pi)                #normalize to 0 to 1
    r, n = np.modf(2*u)  # r = fractional part, n = integer part

    return np.pi*n + np.arccos(1 - 2*r)  #simplied formula, absorbing the 2*pi

def g(x):
    """
    inner to outer
    
    Original fcn
    x = np.asarray(x) % (2*np.pi)  #restrict to 0 to 2pi
    x = x/(2*np.pi)                #normalize to 0 to 1
    n = np.floor(2* x)
    result = (2*np.pi)*((n/2) + 1/2*(np.sin(np.pi*(x - n/2)))**2)
    """
    x = np.asarray(x) % (2*np.pi)

    n = np.floor(x/np.pi)
    r = (x - np.pi*n)/np.pi

    return np.pi*n + np.pi*(1 - np.cos(np.pi*r))/2


def circle_plus(x, y):
    return g(f(x) + f(y))

def circle_minus(x,y):
    return g(f(x) - f(y))




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

    # Compute mean safely
    E = np.full(num_bins, np.nan)
    nonzero = counts > 0
    E[nonzero] = sums[nonzero] / counts[nonzero]

    return E

def mapping(theta):
    y = np.empty_like(theta)

    mask1 = theta <= np.pi
    mask2 = theta > np.pi

    # linear target
    y[mask1] = -1 + 2*theta[mask1]/np.pi
    y[mask2] = 3 - 2*theta[mask2]/np.pi

    delta = np.empty_like(theta)

    # correct branch selection
    delta[mask1] = np.arccos(-y[mask1])                  # [0, π]
    delta[mask2] = 2*np.pi - np.arccos(-y[mask2])        # [π, 2π]

    return delta

# -------------------------------------------------
# Monte Carlo draw
# -------------------------------------------------

lam = np.random.uniform(0, 2*np.pi, N)

a = np.random.uniform(0, 2*np.pi, N)
b = np.random.uniform(0, 2*np.pi, N)


delta_ab = (a-b) % (2*np.pi)

#Delta_ab = campter2ang(ang2campter(a) - ang2campter(b)) % (2*np.pi)

# ------------------------------------
# Signed-measure LHV Monte Carlo
# ------------------------------------
# Local hidden variable angle
lam = np.random.uniform(0, 2*np.pi, N)

# Local deterministic responses
A = np.sign(np.cos(lam - a))
B = np.sign(np.cos(lam + np.pi  - b))


#E_lhv = EXYdelta(A,B,mapping(delta_ab), bins)
E_lhv = EXYdelta(A,B,delta_ab, bins)


#QM theory for testing
A, B = QMtheoryPair(a,b)

#E_QM = EXYdelta(A,B,mapping(delta_ab), bins)
E_QM = EXYdelta(A,B,delta_ab, bins)



# ------------------------------------
# Ideal and Monte Carlo results
# ------------------------------------
delta_fine = np.linspace(0, 2*np.pi, 500)
E_qm_ideal  = -np.cos(delta_fine)
E_lhv_ideal = -2/np.pi * np.arcsin(np.cos(delta_fine))  # triangle


E_mapped = -np.cos(mapping(delta_fine))


#1/np.sqrt(2) * np.cos(delta_fine)  
#correlation is only possible for finite angle sets, such as CHSH

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


#Monte Carlo points
plt.scatter(bin_centers, E_QM,
            color="royalblue",
            s=25,
            alpha=0.8,
            label="QM Monte Carlo")


plt.scatter(bin_centers, E_lhv,
            color="firebrick",
            s=25,
            alpha=0.8,
            label="LHV Monte Carlo")

# plt.scatter(bin_centers, E_ND,
#             color="green",
#             s=25,
#             alpha=0.8,
#             label="Non-Dio Monte Carlo")

plt.xlabel(r"$\delta_{\alpha|\beta}$, difference of Alice and Bob's angle")
plt.ylabel(r"$E(a-b)$")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()