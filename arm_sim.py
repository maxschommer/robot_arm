import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle

import mpl_toolkits.mplot3d.art3d as art3d
import math
import seaborn as sns

class Arm:
    """This holds all of the positions of all of the joints, and the axis of roation of all of the segments""" 

    def __init__(self, rot_axis=[0,0,1], position=[0,0,0]):
        self.rot_axis = [rot_axis]
        self.position = [position]

    def add_joint(self, rot_axis, position):
        self.rot_axis.append(rot_axis)
        self.position.append(position)

    def plot_arm(self, color='b', limits=[-10,10,-10,10,0,10]):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(bottom=(0.05*len(self.position)))
        self.lines = []

        self.ax_joints = []
        self.s_joints = []
        print(len(self.position))
        for i in range(len(self.position)):

            if self.position[i] != [None,None,None]:
                self.lines.insert(0, self.position[i])
            if i != len(self.position)-1:
                self.ax_joints.append(plt.axes([.25, i/20.+.02, 0.65, 0.03], axisbg="w"))
                self.s_joints.append(Slider(self.ax_joints[i], "Base Rotation", -3.14159, 3.14159, valinit=0))
        self.lines_plot, = self.ax.plot(*np.array(self.lines).T)
        print(self.s_joints[0].val)
        
        for i in range(len(self.position)-1):
            self.s_joints[i].on_changed(self.update)

        self.ax.set_xlim3d(limits[0], 10)
        self.ax.set_ylim3d(-10, 10)
        self.ax.set_zlim3d(0, 10)
        

    def update(self, val):
        vals = []
        r_matrix = []
        pos_draw = np.array(self.position)
        pos_draw = pos_draw.astype(float)
        rot_axis_draw = np.array(self.rot_axis)
        rot_axis_draw = rot_axis_draw.astype(float)
  
        for i in range(len(pos_draw)-1):  #Big loop to do all transformations
            if not np.array_equal(pos_draw[i],  np.array([None, None,None])):  #This checks to make sure that None arrays are not subtracted
                for j in range(i+1, len(pos_draw)): #Subtraction Loop that subtracts the first vector in each group from the following items.
                    if not np.array_equal(pos_draw[j],  np.array([None, None,None])): #Checks to make sure that a None vector isn't subtracted from.
                        pos_draw[j] = pos_draw[j]-pos_draw[i]

            rot_matrix = rotation_matrix(rot_axis_draw[i], self.s_joints[i].val) #This is determined by the value of the various sliders
            
            if not np.array_equal(pos_draw[i],  np.array([None, None,None])):  #This checks to make sure that None arrays are not subtracted
                for k in range(i+1, len(rot_axis_draw)):
                    if not np.array_equal(pos_draw[k],  np.array([None, None,None])):
                        temp = np.dot(pos_draw[k], rot_matrix)
                        pos_draw[k] = temp
                        print(pos_draw[k])

            if not np.array_equal(pos_draw[i],  np.array([None, None,None])):  #This checks to make sure that None arrays are not subtracted
                for j in range(i+1, len(pos_draw)): #Subtraction Loop that adds the first vector in each group from the following items.
                    if not np.array_equal(pos_draw[j],  np.array([None, None,None])): #Checks to make sure that a None vector isn't added to.
                        pos_draw[j] = pos_draw[j]+pos_draw[i]
                
            for k in range(i+1, len(rot_axis_draw)):
                rot_axis_draw[k] = np.dot(rot_axis_draw[k], rot_matrix)

        del_index = []
        for p in range(len(pos_draw)):
            if not np.array_equal(pos_draw[p],  np.array([None, None,None])):
                del_index.append(pos_draw[p])
 
        self.ax.cla()
        draw, = self.ax.plot(*np.array(del_index).T)

        self.ax.set_xlim3d(-10, 10)
        self.ax.set_ylim3d(-10, 10)
        self.ax.set_zlim3d(0, 10)
        self.fig.canvas.draw_idle()
    def __str__(self):
        pos_str = str(self.position)
        rot_str = str(self.rot_axis)
        output = 'Position Array: ' + pos_str +'\n'+'Rotation Axis Array: ' + rot_str
        return output

arm = Arm()
arm.add_joint([0,1,0], [0, 0, 0])
arm.add_joint([2,0,6], [2,0,6])
arm.add_joint([0,1,0], [6,0,6])
arm.add_joint([0,1,0], [10,0,8])
arm.add_joint([0,1,0], [13,0,8])
arm.plot_arm()

print(arm)

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

def plot_3D_cylinder(radius, height, fig=arm.fig, elevation=0, resolution=100, color='r', x_center = 0, y_center = 0):
    
    ax = Axes3D(fig, azim=30, elev=30)

    x = np.linspace(x_center-radius, x_center+radius, resolution)
    z = np.linspace(elevation, elevation+height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem

    ax.plot_surface(X, Y, Z, linewidth=0, color=color)
    ax.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")


# params
radius = 3
height = 10
elevation = -5
resolution = 100
color = 'r'
x_center = 0
y_center = 0

plot_3D_cylinder(radius, height, elevation=elevation, resolution=resolution, color=color, x_center=x_center, y_center=y_center)

plt.show()