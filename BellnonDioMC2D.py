# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:12:47 2026

@author: Tom Dunbar
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# def f(x):
#     result = np.arctanh(x/(2*np.pi))
#     return result

# def g(x):
#     result = 2*np.pi*np.tanh(x)
#     return result


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

# ------------------------------------
# Local Deterministic outcome
# ------------------------------------
def LocalDeterm(theta):
    return np.sign(np.cos(theta))

N = 2_500_000        # Monte Carlo samples
nbins = 50           # grid resolution

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
#A = LocalDeterm(lam - a)
#B = LocalDeterm(lam + np.pi  - b)
#z = A*B

# Non-Dio function
A = LocalDeterm(circle_minus(lam,a))
B = LocalDeterm(circle_minus(lam + np.pi,b))
z = A*B



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

plt.title("Monte Carlo Estimate")

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

# minimal style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

ax.set_xlabel("bin centers")
ax.set_ylabel("average z")

plt.title("Diagonal Slice where a + b ≈ 2 pi")


ax.set_ylim([-1.1,1.1])
plt.show()