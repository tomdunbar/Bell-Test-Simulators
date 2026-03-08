# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:09:56 2026

@author: Gamer
"""

import numpy as np
import matplotlib.pyplot as plt


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

fig, ax = plt.subplots()

plt.plot(x, finv(x),
         color="tab:blue",
         linewidth=1,
         linestyle='--',
         label="x vs x'")

plt.plot(x,x,
         color="tab:red",
         linewidth=1,
         label="x vs x")

plt.xlabel(r"x")
plt.ylabel(r"x'")


# ---- Center the axes ----
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide top/right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# ---- Remove grid ----
ax.grid(False)

# ---- Tick marks every 0.5 ----
ax.xaxis.set_major_locator(0.5)
ax.yaxis.set_major_locator(0.5)

# Make ticks appear only on centered axes
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.show()
plt.show()