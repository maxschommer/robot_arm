import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
### create constant z-coordinate
z = np.zeros_like(t)    #               <------------ here
a0 = 5
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)
l, = plt.plot(t, s, lw=2, color='red')
plt.axis([0, 1, -10, 10])

axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg="w")
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg="w")

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


def update(val):
    amp = samp.val
    freq = sfreq.val
    #set constant z coordinate
    l.set_data(t, z)  #               <------------ here
    # set values to y-coordinate
    l.set_3d_properties(amp*np.sin(2*np.pi*freq*t), zdir="y") #<------------ here
    fig.canvas.draw_idle()

sfreq.on_changed(update)
samp.on_changed(update)


axcolor = 'lightgoldenrodyellow'
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()