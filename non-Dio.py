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


x = [0,1/2,1,np.pi,2*np.pi]

xprime = finv(x)

print("x = ",x," and x' = ",xprime,"\n")

xcalc = f(xprime)

print("x' = ",xprime," and x = ",xcalc,"\n")

#delta_fine = np.linspace(0, 2*np.pi, 500)