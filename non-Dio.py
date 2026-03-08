# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:09:56 2026

@author: Gamer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

def f(xprime):
    """
    Non-Diophantine generator (Eq. 57–59 style).
    Vectorized and numerically safe.
    """

    x = np.asarray(xprime)

    n = np.floor(2* xprime)


    x = n/2 + (1/np.pi)*np.arcsin(np.sqrt(2*x - n))

    return x


def finv(x):
    """
    Inverse generator.
    """

    x = np.asarray(x)

    n = np.floor(2*x)
    #print("n  = ", n)

    xprime = (n/2) + 1/2*np.sin(np.pi*(x - n/2))**2

    return xprime

#Test Numbers based on paper
# x = [0,1/2,1,np.pi,2*np.pi]
# xprime = finv(x)
# print("x = ",x," and x' = ",xprime,"\n")
# xcalc = f(xprime)
# print("x' = ",xprime," and x = ",xcalc,"\n")

## Lots of Plots

x = np.linspace(-1, 1, 500)

fig, ax = plt.subplots(figsize=(6,4))


plt.plot(x, finv(x),
         color="tab:blue",
         linewidth=2,
         linestyle='--',
         label="x vs x'")

plt.plot(x,x,
         color="tab:red",
         linewidth=2,
         label="x vs x")

#plt.xlabel(r"x")
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

# ---- tick spacing ----
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# ---- tick label formatting ----
def tick_formatter(x, pos):
    if np.isclose(x % 1, 0):
        return f"{int(x)}"
    elif np.isclose(x % 0.5, 0):
        return f"{x:.1f}"
    return ""

ax.xaxis.set_major_formatter(FuncFormatter(tick_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter))

# ---- professional tick styling ----
ax.tick_params(
    axis='both',
    which='major',
    direction='in',
    length=7,
    width=1,
    top=False,
    right=False
)

ax.tick_params(
    axis='both',
    which='minor',
    direction='in',
    length=4,
    width=0.8,
    top=False,
    right=False
)

# ---- limits ----
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

plt.legend()

plt.tight_layout()
plt.show()