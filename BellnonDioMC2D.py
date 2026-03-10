# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:12:47 2026

@author: Gamer
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:08:41 2026

@author: Gamer
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def f(x):
    result = np.arctanh(x/(2*np.pi))
    return result

def g(x):
    result = 2*np.pi*np.tanh(x)
    return result

N = 2_000_000        # Monte Carlo samples
nbins = 40           # grid resolution

# -------------------------------------------------
# Monte Carlo draw
# -------------------------------------------------

#a = np.random.rand(N)
#b = np.random.rand(N)

a = np.random.uniform(0, 2*np.pi, N)
b = np.random.uniform(0, 2*np.pi, N)

z = g(f(a) + f(b))

# -------------------------------------------------
# Bin edges
# -------------------------------------------------

edges = np.linspace(0,2*np.pi,nbins+1)

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
# Extract bins where a + b = 1 (diagonal)
# -------------------------------------------------

z_diag = np.diag(np.fliplr(z_mean))

# -------------------------------------------------
# 2D Plot of diagonal slice
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(6,4))

ax.scatter(centers, z_diag,s=10)

ax.plot(centers, 2*np.pi*np.ones(len(centers)), color = 'red', linewidth=2, linestyle='--')

# minimal style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

ax.set_xlabel("bin centers")
ax.set_ylabel("average z")

plt.title("Diagonal Slice where a + b ≈ 2 pi")


ax.set_ylim([0,7])
plt.show()