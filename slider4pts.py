# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------------------------------------------------
# mapping function
# --------------------------------------------------

def f(x):
    x = np.asarray(x) % (2*np.pi)

    u = x/(2*np.pi)
    r, n = np.modf(2*u)

    return np.pi*n + np.arccos(1 - 2*r)

# --------------------------------------------------
# parameters
# --------------------------------------------------

r_outer = 1
r_inner = 0.75

alpha0 = 52.2
colors = ["red","red","blue","blue"]

# --------------------------------------------------
# figure
# --------------------------------------------------

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="polar")

t = np.linspace(0,2*np.pi,400)

ax.plot(t, np.full_like(t,r_outer), lw=2)
ax.plot(t, np.full_like(t,r_inner), lw=2)

# styling
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
beta  = alpha - np.deg2rad(22.5)

outer_angles = np.array([
    alpha + np.pi/2,
    alpha - np.pi/2,
    beta + np.pi/2,
    beta - np.pi/2
])

mapped = f(outer_angles)

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
# alpha / beta arrows (from origin)
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

# labels at arrow tips
alpha_label = ax.text(alpha,1.22,"α",color="red",ha="center",va="center")
beta_label  = ax.text(beta,1.22,"β",color="blue",ha="center",va="center")

# --------------------------------------------------
# update
# --------------------------------------------------

def update(alpha_deg):

    alpha = np.deg2rad(alpha_deg)
    beta  = alpha - np.deg2rad(22.5)

    outer_angles = np.array([
        alpha + np.pi/2,
        alpha - np.pi/2,
        beta + np.pi/2,
        beta - np.pi/2
    ])

    mapped = f(outer_angles)

    for i,(a_out,a_in) in enumerate(zip(outer_angles,mapped)):

        outer_pts[i].set_data([a_out],[r_outer])
        inner_pts[i].set_data([a_in],[r_inner])

        radial_lines[i].set_data([a_in,a_in],[0,r_inner])

        arrows[i].set_position((a_out,r_outer))
        arrows[i].xy = (a_in,r_inner)

    # update alpha beta arrows
    alpha_arrow.xy = (alpha,1.15)
    alpha_arrow.set_position((alpha,0))

    beta_arrow.xy = (beta,1.15)
    beta_arrow.set_position((beta,0))

    # update labels
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