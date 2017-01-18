import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
import math

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.25, bottom=0.25)

#Base Coordinate
base = [0,0,0]
should = [2,0,6]

should_len = np.vstack((base, should))
b, = ax.plot(*should_len.T)
prev_line = b


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


ax.set_xlim3d(-10, 10)
ax.set_ylim3d(-10, 10)
ax.set_zlim3d(0, 10)

axrot_base = plt.axes([.25, .2, 0.65, 0.03], axisbg="w")

srot_base = Slider(axrot_base, "Base Rotation", 0, 2*3.14159, valinit=3.14159)


def update(val):
    global prev_line
    
    theta = srot_base.val


    axis_should = [0,0,1]
    should_len = np.dot(rotation_matrix(axis_should,theta), should)
    temp = np.vstack((base, should_len))
    ax.lines.remove(prev_line)
    prev_line, = ax.plot(*temp.T)

    fig.canvas.draw_idle()

srot_base.on_changed(update)

axcolor = 'lightgoldenrodyellow'
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    srot_base.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    b.set_color(label)
    fig.canvas.draw_idle()

radio.on_clicked(colorfunc)

plt.show()