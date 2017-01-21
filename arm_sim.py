import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from numpy import linalg as LA
import mpl_toolkits.mplot3d.art3d as art3d
import math
import seaborn as sns



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
        
        for i in range(len(self.position)-1):
            if not np.array_equal(self.position[i],  self.position[i+1]):
                x, y, z = self.gen_cylinder(self.position[i], self.position[i+1])
                self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')

        print(self.s_joints[0].val)
        
        for i in range(len(self.position)-1):
            self.s_joints[i].on_changed(self.update)

        self.ax.set_xlim3d(-10, 10)
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

        final_draw = []
        for p in range(len(pos_draw)):
            if not np.array_equal(pos_draw[p],  np.array([None, None,None])):
                final_draw.append(pos_draw[p])
 
        self.ax.cla()
        for i in range(len(final_draw)-1):
            if not np.array_equal(final_draw[i],  final_draw[i+1]):
                x, y, z = self.gen_cylinder(final_draw[i], final_draw[i+1])
                self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')

        draw, = self.ax.plot(*np.array(final_draw).T)

        self.ax.set_xlim3d(-10, 10)
        self.ax.set_ylim3d(-10, 10)
        self.ax.set_zlim3d(0, 10)
        self.fig.canvas.draw_idle()


    def gen_cylinder(self, p1=[2,0,6], p2=[6,0,6], r=.75):
        resolution = 15
        #ax = Axes3D(fig, azim=30, elev=30)
        p1 = np.array(p1)
        p2 = np.array(p2)
        diff = p2-p1
        print(diff)
        diff_xy = np.array([diff[0], diff[1]])

        print(diff_xy)
        if diff_xy[1] < 0:
            azimuth = -math.acos(np.inner(diff_xy/LA.norm(diff_xy), np.array([1,0])))
        else:
            azimuth = math.acos(np.inner(diff_xy/LA.norm(diff_xy), np.array([1,0])))
        elevation = math.acos(np.inner(diff/LA.norm(diff), np.array([0,0,1])))
        print(azimuth)
        print(elevation)
        diff = p1-p2
        mag = math.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        h = np.linspace(0,mag, resolution)
        x = r*np.outer(np.cos(u), np.ones(np.size(v)))
        y = r*np.outer(np.sin(u), np.ones(np.size(v)))
        z = np.outer(np.ones(np.size(u)), h)
        for i in range(resolution):
            for j in range(resolution):
                point = np.array([x[i][j], y[i][j], z[i][j]])
                rot_matrix = rotation_matrix([0,0,1], azimuth) #This is determined by the value of the various sliders
                rot_axis = np.dot(rot_matrix, np.array([0,1,0]))
                rot_matrix_f = rotation_matrix(rot_axis, elevation)
                point = np.dot(rot_matrix_f, point)
                x[i][j] = point[0]+p1[0]
                y[i][j] = point[1]+p1[1]
                z[i][j] = point[2]+p1[2]

        return x, y, z

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
plt.show()