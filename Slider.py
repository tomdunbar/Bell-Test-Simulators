import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ----------------------------
# mapping function
# ----------------------------

def f(x):
    x = np.asarray(x) % (2*np.pi)

    u = x/(2*np.pi)
    r, n = np.modf(2*u)

    return np.pi*n + np.arccos(1 - 2*r)

# ----------------------------
# geometry
# ----------------------------

r_outer = 1
r_inner = 0.75
N = 40

base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)

# ----------------------------
# figure
# ----------------------------

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')

t = np.linspace(0, 2*np.pi, 400)

ax.plot(t, np.full_like(t, r_outer), lw=2)
ax.plot(t, np.full_like(t, r_inner), lw=2)

outer_pts, = ax.plot([], [], 'o')
mapped_pts, = ax.plot([], [], 'o')

arrows = []


ax.spines['polar'].set_visible(False)

ax.grid(True)
ax.set_theta_zero_location("E")
ax.set_thetagrids([0,90,180,270])
ax.xaxis.grid(True, alpha=0.3)


ax.yaxis.grid(False)
ax.set_yticklabels([])
ax.set_ylim([0,1.2])

# ----------------------------
# update function
# ----------------------------

def update(theta_deg):

    global arrows

    theta = np.deg2rad(theta_deg)

    outer_angles = (base_angles + theta) % (2*np.pi)
    mapped_angles = f(outer_angles)

    outer_pts.set_data(outer_angles, np.full_like(outer_angles, r_outer))
    mapped_pts.set_data(mapped_angles, np.full_like(mapped_angles, r_inner))

    # remove old arrows
    for a in arrows:
        a.remove()
    arrows = []

    # draw arrows
    for a1, a2 in zip(outer_angles, mapped_angles):

        arrow = ax.annotate(
            "",
            xy=(a2, r_inner),
            xytext=(a1, r_outer),
            arrowprops=dict(arrowstyle="->", lw=1)
        )

        arrows.append(arrow)

    fig.canvas.draw_idle()

# ----------------------------
# slider
# ----------------------------

slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.03])

slider = Slider(
    slider_ax,
    "Rotation θ (deg)",
    0,
    360,
    valinit=0
)

slider.on_changed(update)

# initial draw
update(0)



plt.show()