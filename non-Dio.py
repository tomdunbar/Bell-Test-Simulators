# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:09:56 2026

@author: Tom Dunbar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

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

def circle_mul(x,y):
    return g(f(x) * f(y))

def circle_div(x,y):
    fy = f(y)
    fy = np.where(np.abs(fy) < 1e-10, np.nan, fy)
    return g(f(x) / fy)

#Test Numbers based on paper
# x = [0,1/2,1,np.pi,2*np.pi]
# xprime = g(x)
# print("x = ",x," and x' = ",xprime,"\n")
# xcalc = f(xprime)
# print("x' = ",xprime," and x = ",xcalc,"\n")

# -------------------------------------------------
# parameters
# -------------------------------------------------

r1 = 0.75
r2 = 1
N = 60

theta = np.linspace(0, 2*np.pi, N, endpoint=False)

theta_transformed = f(theta)

# -------------------------------------------------
# plot circle mapping via f(x)
# -------------------------------------------------

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')

# circles
t = np.linspace(0, 2*np.pi, 400)
ax.plot(t, np.full_like(t, r2), linewidth=1, color="black")
ax.plot(t, np.full_like(t, r1), linewidth=1, color="black")

# connecting lines
# for i in range(N):
#     ax.plot(
#         [theta[i], theta_transformed[i]],
#         [r2, r1],
#         linewidth=1,
#         alpha=0.7,
#         color="grey"
#     )

# connecting arrows
for i in range(N):
    ax.annotate(
        "",
        xy=(theta_transformed[i], r1),   # arrow tip
        xytext=(theta[i], r2),           # arrow base
        arrowprops=dict(
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=.8,
        alpha=0.7,
        color="grey"
        ),
    )


# outer points
ax.scatter(theta, np.full_like(theta, r2), s=50)

# transformed points
ax.scatter(theta_transformed, np.full_like(theta_transformed, r1), s=50)

# minimal style
ax.spines['polar'].set_visible(False)

ax.grid(True)
ax.set_theta_zero_location("E")
ax.set_thetagrids([0,90,180,270])
ax.xaxis.grid(True, alpha=0.3)
#ax.set_xticklabels([])

ax.yaxis.grid(False)
ax.set_yticklabels([])
ax.set_ylim([0,1.2])

plt.title("Circle Mapping via f(x)")
plt.show()

# -------------------------------------------------
# plot circle mapping via g(x)
# -------------------------------------------------

theta_transformed = g(theta)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')

# circles
t = np.linspace(0, 2*np.pi, 400)
ax.plot(t, np.full_like(t, r2), linewidth=1, color="black")
ax.plot(t, np.full_like(t, r1), linewidth=1, color="black")

# connecting arrows
for i in range(N):
    ax.annotate(
        "",
        xy=(theta_transformed[i], r2),   # arrow tip
        xytext=(theta[i], r1),           # arrow base
        arrowprops=dict(
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=.8,
        alpha=0.7,
        color="grey"
        ),
    )


# inner points
ax.scatter(theta, np.full_like(theta, r1), s=50, color = 'red')

# transformed points
ax.scatter(theta_transformed, np.full_like(theta_transformed, r2), s=50, color = 'purple')

# minimal style
ax.spines['polar'].set_visible(False)

ax.grid(True)
ax.set_theta_zero_location("E")
ax.set_thetagrids([0,90,180,270])
ax.xaxis.grid(True, alpha=0.3)
#ax.set_xticklabels([])

ax.yaxis.grid(False)
ax.set_yticklabels([])
ax.set_ylim([0,1.2])

plt.title("Circle Mapping via g(x)")
plt.show()



# -------------------------------------------------
# Cartesian plot confirming f(x),  g(x), and g(f(x))
# -------------------------------------------------

x = np.linspace(0, 2*np.pi, 500, endpoint=False)

fig, ax = plt.subplots(figsize=(6,4))


# ---- limits ----
ax.set_xlim(0,2*np.pi)
ax.set_ylim(0,2*np.pi)

#--- Plots----
plt.plot(x, f(x),
         color="tab:blue",
         linewidth=2,
         linestyle='--',
         label="f(x)")

plt.plot(x,g(x),
         color="tab:red",
         linewidth=2,
         label="g(x)")

plt.plot(x,g(f(x)),
         color="tab:green",
         linewidth=2,
         label="g(f(x))")

plt.xlabel(r"x")
#plt.ylabel(r"x'")

# Journal-style parameters
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.linewidth": 1.2,
    "font.size": 12,
})

# ---- center axes ----
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


Piticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
Pitick_labels = [
    "0",
    r"$\frac{\pi}{2}$",
    r"$\pi$",
    r"$\frac{3\pi}{2}$",
    r"$2\pi$"
]

ax.set_xticks(Piticks)
ax.set_xticklabels(Pitick_labels)

ax.set_yticks(Piticks)
ax.set_yticklabels(Pitick_labels)

plt.legend()

plt.tight_layout()
plt.show()


# # ---- tick spacing ----
# ax.xaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_major_locator(MultipleLocator(0.5))

# ax.xaxis.set_minor_locator(MultipleLocator(0.1))
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# # ---- tick label formatting ----
# def tick_formatter(x, pos):
#     if np.isclose(x % 1, 0):
#         return f"{int(x)}"
#     elif np.isclose(x % 0.5, 0):
#         return f"{x:.1f}"
#     return ""

# ax.xaxis.set_major_formatter(FuncFormatter(tick_formatter))
# ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter))

# # ---- professional tick styling ----
# ax.tick_params(
#     axis='both',
#     which='major',
#     direction='in',
#     length=7,
#     width=1,
#     top=False,
#     right=False
# )

# ax.tick_params(
#     axis='both',
#     which='minor',
#     direction='in',
#     length=4,
#     width=0.8,
#     top=False,
#     right=False
# )





# ----
# Origional Model

# def f(xprime):
#     """
#     Non-Diophantine generator (Eq. 57–59 style).
#     Vectorized
#     """
#     xprime = np.asarray(xprime)

#     n = np.floor(2* xprime)

#     x = n/2 + (1/np.pi)*np.arcsin(np.sqrt(2*xprime - n))

#     return x


# def finv(x):
#     """
#     Inverse generator.
#     """
#     x = np.asarray(x)

#     n = np.floor(2*x)

#     xprime = (n/2) + 1/2*np.sin(np.pi*(x - n/2))**2

#     return xprime


# def circle_plus(x,y):
#     return finv(f(x) + f(y))

# def circle_minus(x,y):
#     return finv(f(x) - f(y))

# def circle_mul(x,y):
#     return finv(f(x) * f(y))

# def circle_div(x,y):
#     return finv(f(x) / f(y))
