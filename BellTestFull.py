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
#aPlusb = (a + b) % (2*np.pi)

# ------------------------------------
# Signed-measure LHV Monte Carlo
# ------------------------------------
# Local hidden variable angle
lam = np.random.uniform(0, 2*np.pi, N)

# Local deterministic responses
A = np.sign(np.cos(lam - a))
B = np.sign(np.cos(lam + np.pi  - b))

E_lhv = EXYdelta(A,B,delta_ab, bins)

# ------------------------------------
# local maximum correlation (LMV) Monte Carlo
# ------------------------------------
num_points = 10000   # discretization for the PDF
phi = np.linspace(0, np.pi, num_points)
halfsin_pdf = 0.5 * np.sin(phi)
halfsin_pdf /= np.sum(halfsin_pdf)   # normalize
halfsin_samples = np.random.choice(phi, size=N, p=halfsin_pdf)

A_lmc = np.sign(np.cos(a))
B_lmc = -np.sign(np.cos(b - halfsin_samples + np.pi/2))

E_lmc = EXYdelta(A_lmc,B_lmc,delta_ab, bins)
#this is off by a factor of pi/2, but its the closest cosine curve we can get
#The correlation becomes the overlap of two shifted square waves → linear in θ.
#Fourier expanding that linear function gives first harmonic amplitude: 2/pi

# ------------------------------------
# 1/2 cosine LHV Monte Carlo
# ------------------------------------
# phi = np.linspace(0, np.pi, num_points)
# scaledCos_pdf =  0.5 * np.sin(phi) + np.pi/4
# #scaledCos_pdf =  0.5 * np.sin(phi) + (np.pi-1)/2
# scaledCos_pdf /= np.sum(scaledCos_pdf)   # normalize
# scaledCos_samples = np.random.choice(phi, size=N, p=scaledCos_pdf)

# A_sc = np.sign(np.cos(a))
# B_sc = -np.sign(np.cos(b - scaledCos_samples + np.pi/2))

# E_sc = EXYdelta(A_sc,B_sc,delta_ab, bins)


# ------------------------------------
# Quantum singlet outcomes Monte Carlo
# ------------------------------------
X,Y = QMtheoryPair(a, b)

E_QM = EXYdelta(X,Y,delta_ab , bins)

# ------------------------------------
# Ideal and Monte Carlo results
# ------------------------------------
delta_fine = np.linspace(0, 2*np.pi, 500)
E_qm_ideal  = -np.cos(delta_fine)
E_lhv_ideal = -2/np.pi * np.arcsin(np.cos(delta_fine))  # triangle
E_lmc_ideal = -2/np.pi * np.cos(delta_fine)  #local max correlation


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

plt.plot(delta_fine, E_lmc_ideal,
         color="tab:green",
         linewidth=2.5,
         label="LMC model")

# Monte Carlo points
plt.scatter(bin_centers, E_QM,
            color="royalblue",
            s=25,
            alpha=0.8,
            label="QM Monte Carlo")

plt.scatter(bin_centers, E_lhv,
            color="firebrick",
            s=25,
            alpha=0.8,
            label="Typical LHV Monte Carlo")

plt.scatter(bin_centers, E_lmc,
            color="green",
            s=25,
            alpha=0.8,
            label="Local Maximum Correlation Monte Carlo")

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

