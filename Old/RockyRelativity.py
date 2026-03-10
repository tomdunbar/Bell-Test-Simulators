import numpy as np
import matplotlib.pyplot as plt


def EXYdelta(X, Y, delta, bins):

    bin_indices = np.digitize(delta, bins) - 1
    num_bins = len(bins) - 1

    valid = (bin_indices >= 0) & (bin_indices < num_bins)

    bin_indices = bin_indices[valid]
    XY = (X * Y)[valid]

    counts = np.bincount(bin_indices, minlength=num_bins)
    sums = np.bincount(bin_indices, weights=XY, minlength=num_bins)

    E = np.full(num_bins, np.nan)
    nonzero = counts > 0
    E[nonzero] = sums[nonzero] / counts[nonzero]

    return E


# Number of samples
N = 200000

# velocities (fractions of c)
a = np.random.rand(N)
b = np.random.rand(N)

# "Prediction" (classical expectation)
def predicted_velocity(a, b):
    return a + b

# "Experimental result"
def experiment_velocity(a, b):
    return (a + b) / (1 + a*b)


v_pred = predicted_velocity(a, b)
v_exp = experiment_velocity(a, b)

delta = a + b

# bins of a+b
bins = np.linspace(0, 2, 200)
delta_centers = 0.5 * (bins[:-1] + bins[1:])

X = np.ones_like(delta)

E_pred = EXYdelta(X, v_pred, delta, bins)
E_exp = EXYdelta(X, v_exp, delta, bins)

# plot
plt.figure(figsize=(8,6))

plt.plot(delta_centers, E_pred, label="Predicted velocity (classical)", linewidth=3)
plt.plot(delta_centers, E_exp, label="Experiment", linewidth=3)

plt.xlabel("a + b")
plt.ylabel("Resulting velocity")
plt.title("Velocity Addition: Prediction vs Experiment")

plt.legend()
plt.grid(True)

plt.show()



############# QM
delta_fine = np.linspace(0, np.pi, 500)

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