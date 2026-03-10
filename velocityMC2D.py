# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:08:41 2026

@author: Gamer
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------
# Parameters
# -------------------------------------------------

N = 2_000_000        # Monte Carlo samples
nbins = 40           # grid resolution

# -------------------------------------------------
# Monte Carlo draw
# -------------------------------------------------

a = np.random.rand(N)
b = np.random.rand(N)

z = np.tanh(np.arctanh(a) + np.arctanh(b))

# -------------------------------------------------
# Bin edges
# -------------------------------------------------

edges = np.linspace(0,1,nbins+1)

# Sum of z in each bin
z_sum, _, _ = np.histogram2d(a, b, bins=[edges,edges], weights=z)

# Count samples per bin
counts, _, _ = np.histogram2d(a, b, bins=[edges,edges])

# Mean z per bin
z_mean = z_sum / counts
z_mean[counts == 0] = np.nan

# -------------------------------------------------
# Bin centers
# -------------------------------------------------

centers = 0.5*(edges[:-1] + edges[1:])
A,B = np.meshgrid(centers,centers)

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    A.flatten(),
    B.flatten(),
    z_mean.flatten(),
    s=12,
    alpha=0.9
)

# minimal / Tufte style
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlabel("avg z")

plt.title("Monte Carlo Estimate of z = tanh(artanh(a)+artanh(b))")

ax.set_zlabel("avg z")


plt.show()

# -------------------------------------------------
# Extract bins where a ≈ b (diagonal)
# -------------------------------------------------

z_diag = np.diag(z_mean)

# -------------------------------------------------
# 2D Plot of diagonal slice
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(6,4))

ax.scatter(centers, z_diag,s=10)

ax.plot(centers, 2*centers, color = 'red', linewidth=2)

# minimal style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

ax.set_xlabel("a = b (bin centers)")
ax.set_ylabel("average z")

plt.title("Monte Carlo slice where a ≈ b")

plt.show()