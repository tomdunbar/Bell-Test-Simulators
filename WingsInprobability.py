# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:55:18 2026

@author: Gamer
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------------------------------------------------
# nonlinear mapping (original mapping)
# --------------------------------------------------

# def g(x):
#     x = np.asarray(x) % (2*np.pi)

#     u = x/(2*np.pi)
#     r, n = np.modf(2*u)

#     return np.pi*n + np.arccos(1 - 2*r)


# --------------------------------------------------
# sinusoidal deformation
# g(x) = π/4 sin(2x)
# --------------------------------------------------

def g(x):
     return (np.pi/3) * np.sin(2*x)

#def g(x):
#    return -(1/2) * np.arccos(np.sin(2*x)) + np.pi/4


# --------------------------------------------------
# measurement geometry
# --------------------------------------------------

def limit_angles(alpha, beta):

    a1 = alpha + np.pi/2 - g(alpha)
    a2 = alpha - np.pi/2 + g(alpha)

    b1 = beta + np.pi/2 - g(beta)
    b2 = beta - np.pi/2 + g(beta)

    return np.array([a1, a2, b1, b2])


# --------------------------------------------------
# circular angle difference
# --------------------------------------------------

def angdiff(a, b):
    return np.abs(np.arctan2(np.sin(a-b), np.cos(a-b)))


def cw_angle(x, y):
    theta = (x - y) % (2*np.pi)
    return np.where(theta > np.pi, 0, theta)

def ccw_angle(x, y):
    theta = (y - x) % (2*np.pi)
    return np.where(theta > np.pi, 0, theta)

# --------------------------------------------------
# probability calculation
# --------------------------------------------------

def probability(alpha, beta):

    a1, a2, b1, b2 = limit_angles(alpha, beta)

    d1 = cw_angle(a1, b1)
    d2 = ccw_angle(a2, b2)

    return (d1 + d2) / (2*np.pi)


# --------------------------------------------------
# parameters
# --------------------------------------------------

r_outer = 1
r_inner = 0.75

alpha0 = 0
delta_deg = 22.5

delta = np.deg2rad(delta_deg)

colors = ["red","red","blue","blue"]


# --------------------------------------------------
# numerical average
# --------------------------------------------------

alpha_vals = np.linspace(0,2*np.pi,100000)
beta_vals = alpha_vals - delta

vals = probability(alpha_vals, beta_vals)

p = np.mean(vals)

print("numerical average:", p)
print("target:", (1-np.cos(delta))/2)


# --------------------------------------------------
# figure
# --------------------------------------------------

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="polar")

t = np.linspace(0,2*np.pi,400)

ax.plot(t, np.full_like(t,r_outer), lw=2)
ax.plot(t, np.full_like(t,r_inner), lw=2)

ax.spines['polar'].set_visible(False)

ax.grid(True)

ax.set_theta_zero_location("E")
ax.set_thetagrids([0,90,180,270])

ax.xaxis.grid(True, alpha=0.3)

ax.yaxis.grid(False)
ax.set_yticklabels([])

ax.set_ylim([0,1.3])


# --------------------------------------------------
# initial geometry
# --------------------------------------------------

alpha = np.deg2rad(alpha0)
beta = alpha - delta

outer_angles = [alpha + np.pi/2, alpha - np.pi/2, beta + np.pi/2,beta- np.pi/2]
mapped = limit_angles(alpha, beta)


# --------------------------------------------------
# artists
# --------------------------------------------------

outer_pts = []
inner_pts = []
radial_lines = []
arrows = []

for a_out, a_in, c in zip(outer_angles, mapped, colors):

    p1, = ax.plot(a_out, r_outer, "o", color=c)
    outer_pts.append(p1)

    p2, = ax.plot(a_in, r_inner, "o", color=c)
    inner_pts.append(p2)

    line, = ax.plot([a_in,a_in],[0,r_inner], color=c)
    radial_lines.append(line)

    arrow = ax.annotate(
        "",
        xy=(a_in,r_inner),
        xytext=(a_out,r_outer),
        arrowprops=dict(arrowstyle="->", color=c)
    )

    arrows.append(arrow)


# --------------------------------------------------
# alpha / beta arrows
# --------------------------------------------------

alpha_arrow = ax.annotate(
    "",
    xy=(alpha,1.15),
    xytext=(alpha,0),
    arrowprops=dict(arrowstyle="->", linestyle="--", color="red", lw=2)
)

beta_arrow = ax.annotate(
    "",
    xy=(beta,1.15),
    xytext=(beta,0),
    arrowprops=dict(arrowstyle="->", linestyle="--", color="blue", lw=2)
)

alpha_label = ax.text(alpha,1.22,"α",color="red",ha="center",va="center")
beta_label  = ax.text(beta,1.22,"β",color="blue",ha="center",va="center")


# --------------------------------------------------
# update function
# --------------------------------------------------

def update(alpha_deg):

    alpha = np.deg2rad(alpha_deg)
    beta = alpha - delta

    outer_angles = [alpha + np.pi/2, alpha - np.pi/2, beta + np.pi/2,beta- np.pi/2]
    mapped = limit_angles(alpha, beta)

    for i,(a_out,a_in) in enumerate(zip(outer_angles,mapped)):

        outer_pts[i].set_data([a_out],[r_outer])
        inner_pts[i].set_data([a_in],[r_inner])

        radial_lines[i].set_data([a_in,a_in],[0,r_inner])

        arrows[i].set_position((a_out,r_outer))
        arrows[i].xy = (a_in,r_inner)

    alpha_arrow.xy = (alpha,1.15)
    alpha_arrow.set_position((alpha,0))

    beta_arrow.xy = (beta,1.15)
    beta_arrow.set_position((beta,0))

    alpha_label.set_position((alpha,1.22))
    beta_label.set_position((beta,1.22))

    fig.canvas.draw_idle()


# --------------------------------------------------
# slider
# --------------------------------------------------

slider_ax = fig.add_axes([0.2,0.05,0.6,0.03])

slider = Slider(
    slider_ax,
    "alpha (deg)",
    0,
    360,
    valinit=alpha0
)

slider.on_changed(update)

plt.show()


# --------------------------------------------------
# second plot
# --------------------------------------------------

alpha_vals = np.linspace(0,2*np.pi,1000)
beta_vals = alpha_vals - delta

vals = probability(alpha_vals, beta_vals)

avg_val = np.mean(vals)
qm_val = (1 - np.cos(delta)) / 2

fig2, ax2 = plt.subplots(figsize=(7,4))

ax2.plot(alpha_vals, vals, lw=2, label="calculated")

ax2.axhline(
    avg_val,
    linestyle="--",
    lw=2,
    label=f"average = {avg_val:.5f}"
)

ax2.axhline(
    qm_val,
    linestyle=":",
    lw=2,
    color="red",
    label=f"(1-cos({delta_deg}°))/2 = {qm_val:.5f}"
)

ax2.set_xlim(0,2*np.pi)

xticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
xtick_labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

ax2.set_xticks(xticks)
ax2.set_xticklabels(xtick_labels)

ax2.set_xlabel(r"$\alpha$")
ax2.set_ylabel("value")

ax2.set_title("Computed probability vs α")

ax2.legend()

ax2.grid(True, alpha=0.3)

plt.show()

# --------------------------------------------------
# third plot: average probability vs delta
# --------------------------------------------------

# delta grid
delta_vals = np.linspace(0, 2*np.pi, 200)

# integration grid
alpha_vals = np.linspace(0, 2*np.pi, 4000)

# preallocate (faster)
avg_probs = np.zeros_like(delta_vals)

# compute averages
for i, delta in enumerate(delta_vals):

    beta_vals = alpha_vals - delta

    vals = probability(alpha_vals, beta_vals)

    avg_probs[i] = np.mean(vals)


# --------------------------------------------------
# reference curves
# --------------------------------------------------

# QM probability of SAME result
qm_curve = -1/2 *(1 + np.cos(delta_vals)) / 2 + 1/2

# triangle wave between 0 and 1
triangle_curve = -1/2*np.abs((delta_vals % (2*np.pi)) - np.pi) / np.pi +1/2


# --------------------------------------------------
# plot
# --------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(7,4))

ax3.plot(delta_vals, avg_probs,
         lw=2,
         label="model")

ax3.plot(delta_vals, qm_curve,
         lw=2,
         linestyle="--",
         label="QM same probability")

ax3.plot(delta_vals, triangle_curve,
         lw=2,
         linestyle=":",
         label="triangle δ")

# axis formatting
ax3.set_xlim(0,2*np.pi)

xticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
xtick_labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

ax3.set_xticks(xticks)
ax3.set_xticklabels(xtick_labels)

ax3.set_xlabel(r"$\delta$")
ax3.set_ylabel("average probability")

ax3.set_title("Average probability vs δ")

ax3.legend()

ax3.grid(True, alpha=0.3)

plt.show()