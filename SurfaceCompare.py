import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# Helper styling
# =========================================================

def tufte_3d(ax):
    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.line.set_color("0.7")
    ax.yaxis.line.set_color("0.7")
    ax.zaxis.line.set_color("0.7")


def tufte_2d(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


# =========================================================
# Plot 1
# =========================================================

n = 200
a = np.linspace(0,0.999,n)
b = np.linspace(0,0.999,n)

A,B = np.meshgrid(a,b)

Z1 = A + B
Z2 = np.tanh(np.arctanh(A) + np.arctanh(B))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(A,B,Z1,alpha=0.45,linewidth=0)
ax.plot_surface(A,B,Z2,alpha=0.45,linewidth=0)

ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlabel("z")

ax.set_xticks([1])
ax.set_yticks([1])
ax.set_zticks([1,2])

tufte_3d(ax)

plt.title("Surface Comparison")
plt.show()


# =========================================================
# Plot 2  (intersection with plane a=b)
# =========================================================

a = np.linspace(0,0.999,500)
b = a

z1 = a + b
z2 = np.tanh(np.arctanh(a) + np.arctanh(b))

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(a, z1, linewidth=2)
ax.plot(a, z2, linewidth=2)

ax.set_xlabel("a = b")
ax.set_ylabel("z")

tufte_2d(ax)

plt.title("Intersection with plane a=b")
plt.show()

# =========================================================
# Plot 3
# =========================================================

n = 200

a = np.linspace(-np.pi, np.pi, n)
b = np.linspace(-np.pi, np.pi, n)

A,B = np.meshgrid(a,b)

Z3 = -np.cos(A-B)
Z4 = -2/np.pi * np.arcsin(np.cos(np.mod(A-B,2*np.pi)))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(A,B,Z3,alpha=0.45,linewidth=0)
ax.plot_surface(A,B,Z4,alpha=0.45,linewidth=0)

ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlabel("z")

tufte_3d(ax)

plt.title("Periodic Surface Comparison")
plt.show()


# =========================================================
# Plot 4 (slice a = -b)
# =========================================================

a = np.linspace(-np.pi, np.pi, 600)
b = -a

z3 = -np.cos(a-b)
z4 = -2/np.pi * np.arcsin(np.cos(np.mod(a-b,2*np.pi)))

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(a, z3, linewidth=2)
ax.plot(a, z4, linewidth=2)

ax.set_xlabel("a   (b = -a)")
ax.set_ylabel("z")

tufte_2d(ax)

plt.title("Slice where a = -b")
plt.show()