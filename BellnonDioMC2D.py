# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:12:47 2026

@author: Tom Dunbar
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

def LocalDeterm(theta):
    return np.sign(np.cos(theta))

N = 3_000_000        # Monte Carlo samples
nbins = 60           # grid resolution

# -------------------------------------------------
# Monte Carlo draw
# -------------------------------------------------

lam = np.random.uniform(0, 2*np.pi, N)

a = np.random.uniform(0, 2*np.pi, N)
b = np.random.uniform(0, 2*np.pi, N)

#QM theory for testing
#A, B = QMtheoryPair(a,b)
#z = A*B

# Local deterministic responses for testing
A = LocalDeterm(lam - a)
B = LocalDeterm(lam + np.pi  - b)
z = A*B

# Non-Dio function
#z = g(f(a) + f(b))



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
# ideal -cos(delta) surface
# -------------------------------------------------
k = 200
m = np.linspace(0,0.999,k)*2*np.pi
n = np.linspace(0,0.999,k)*2*np.pi

M,N = np.meshgrid(m,n)

#Z_ideal = g(f(M) + f(N))

Z_ideal = -np.cos(M-N)

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(M,N,Z_ideal,
                color="red",
                alpha=0.35,
                linewidth=0)

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

plt.title("Monte Carlo Estimate of z = g(f(a)+f(b))")

ax.set_zlabel("avg z")

ax.view_init(elev=2, azim=225)

plt.show()

# -------------------------------------------------
# Extract bins where a + b = 2pi (anti-diagonal)
# -------------------------------------------------

z_diag = np.diag(np.fliplr(z_mean))

# -------------------------------------------------
# 2D Plot of diagonal slice
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(centers, z_diag, marker='o', markersize=5, linestyle='-')


ax.plot(m, np.diag(np.fliplr(Z_ideal)),color = 'red', linewidth=2)
#ax.plot(centers, -np.cos(2*centers), color = 'red', linewidth=2, linestyle='--')
#ax.plot(centers, 2*np.pi*np.ones(len(centers)), color = 'red', linewidth=2, linestyle='--')

# minimal style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

ax.set_xlabel("bin centers")
ax.set_ylabel("average z")

plt.title("Diagonal Slice where a + b ≈ 2 pi")


ax.set_ylim([-1.1,1.1])
plt.show()