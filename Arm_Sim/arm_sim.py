import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from numpy import linalg as LA
import mpl_toolkits.mplot3d.art3d as art3d
import math
import random
import seaborn as sns
from scipy import stats
from mayavi import mlab
import multiprocessing
from vector import Vector


class Arm:
	"""This holds all of the positions of all of the joints, and the axis of roation of all of the segments""" 

	def __init__(self, rot_axis=[0,0,1], position=[0,0,0]):
		self.rot_axis = [rot_axis]
		self.position = [position]
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection="3d")
		self.lines = []
		self.ax_joints = []
		self.s_joints = []


	def add_joint(self, rot_axis, position):
		self.rot_axis.append(rot_axis)
		self.position.append(position)


	def plot_arm(self, limits=[-10,10,-10,10,0,10], Plot=True):
		color = 'r' if self.is_self_intersecting(self.position) else 'b'

		plt.subplots_adjust(bottom=(0.05*len(self.position)))

		for i in range(len(self.position)):

			if self.position[i] != [None,None,None]:
				self.lines.insert(0, self.position[i])
			if i != len(self.position)-1:
				self.ax_joints.append(plt.axes([.25, i/20.+.02, 0.65, 0.03], axisbg="w"))
				self.s_joints.append(Slider(self.ax_joints[i], "Base Rotation", -3.14159, 3.14159, valinit=0))

		if Plot:
			self.lines_plot, = self.ax.plot(*np.array(self.lines).T)
	   
			for i in range(len(self.position)-1):
				if not np.array_equal(self.position[i],  self.position[i+1]):
					x, y, z = gen_cylinder(self.position[i], self.position[i+1])
					self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color)
					x, y, z = gen_sphere(self.position[i+1])
					self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color='k')

		for i in range(len(self.position)-1):
			self.s_joints[i].on_changed(self.update)

		self.ax.set_xlim3d(-10, 10)
		self.ax.set_ylim3d(-10, 10)
		self.ax.set_zlim3d(0, 10)


	def update(self, val):
		final_draw = self.get_arm_pos()
		color = 'r' if self.is_self_intersecting(final_draw) else 'b'
		self.ax.cla()
		for i in range(len(final_draw)-1):
			if not np.array_equal(final_draw[i],  final_draw[i+1]):
				x, y, z = gen_cylinder(final_draw[i], final_draw[i+1])
				self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color)
				x, y, z = gen_sphere(final_draw[i+1])
				self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color='k')
		draw, = self.ax.plot(*np.array(final_draw).T)

		self.ax.set_xlim3d(-10, 10)
		self.ax.set_ylim3d(-10, 10)
		self.ax.set_zlim3d(0, 10)
		self.fig.canvas.draw_idle()


	def get_arm_pos(self):
		"""
		Returns a list. Not a numpy array, contrary to what Schommer will tell you
		"""
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

			if not np.array_equal(pos_draw[i],  np.array([None, None,None])):  #This checks to make sure that None arrays are not subtracted
				for j in range(i+1, len(pos_draw)): #Subtraction Loop that adds the first vector in each group from the following items.
					if not np.array_equal(pos_draw[j],  np.array([None, None,None])): #Checks to make sure that a None vector isn't added to.
						pos_draw[j] = pos_draw[j]+pos_draw[i]
				
			for k in range(i+1, len(rot_axis_draw)):
				rot_axis_draw[k] = np.dot(rot_axis_draw[k], rot_matrix)

		final_draw = []
		for p in range(len(pos_draw)):
			if not np.array_equal(pos_draw[p],  np.array([None,None,None])):
				final_draw.append(pos_draw[p])

		return final_draw


	def jacobian(self, delta=0.01):
		"""
		Make a totally real and exactly calculated with lots of algebra and trig jacobian matrix
		"""
		matrix = np.zeros((3, len(self.s_joints)))
		position0 = np.array(self.get_arm_pos()[-1])
		for i, s in enumerate(self.s_joints):
			s.val += delta
			position1 = np.array(self.get_arm_pos()[-1])
			matrix[:,i] = (position1-position0)/delta
			s.val -= delta

		return matrix


	def move_to(self, final_position, step=1):
		"""
		Move to the new position. Return 0 for success and 1 for failure
		"""
		if self.is_self_intersecting():
			print("ERR: I am dead.")
			return 0

		current_position = np.array(self.get_arm_pos()[-1])
		tot_dist = np.linalg.norm(final_position-current_position)
		tot_steps = int(tot_dist/step)
		for i in range(tot_steps):
			DF = self.jacobian()
			while True:
				if not DF.any():
					print("ERR: I am stuck")
					return 0
				current_position = np.array(self.get_arm_pos()[-1])
				current_step = (final_position-current_position)/(tot_steps-i)
				commands = np.dot(np.linalg.pinv(DF),current_step)
				for j, command in enumerate(commands):
					self.s_joints[j].val += command
					if self.is_self_intersecting():
						self.s_joints[j].val -= command
						DF[:,j] = 0
						break

				break
		return 1


	def straighten(self):
		"""
		Return an equivalent arm pointing straight up
		"""
		pass


	"""Work in progress"""
	def parameter_sweep(self, index=0, iterations=10000):
		random.seed(5)
		X=[]
		Y=[]
		Z=[]
		for i in range(0, iterations):
			for j in range(0, len(self.s_joints)):
				self.s_joints[j].val = random.uniform(self.s_joints[j].valmin, self.s_joints[j].valmax)
			pose = self.get_arm_pos()
			point = pose[-1]
			if not self.is_self_intersecting(position_arr=pose):
				X.append(point[0])
				Y.append(point[1])
				Z.append(point[2])
		print(X)
		x = np.array(X)
		y = np.array(Y)	
		z = np.array(Z)

		xyz = np.vstack([x,y,z])
		kde = stats.gaussian_kde(xyz)

		# Evaluate kde on a grid
		xmin, ymin, zmin = x.min(), y.min(), z.min()
		xmax, ymax, zmax = x.max(), y.max(), z.max()
		xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
		coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
		density = kde(coords).reshape(xi.shape)

		# Plot scatter with mayavi
		figure = mlab.figure('DensityPlot')

		grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
		min = density.min()
		max=density.max()
		mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))

		mlab.axes()
		mlab.show()
		#points, sub = self.hist3d_bubble(X, Y, Z, bins=4)
		#points = self.recursive_layer(dtheta=dtheta, voxel_dim=voxel_dim, index=index, sweep=sweep)
		#self.ax.scatter(X, Y, Z)

   
	def __str__(self):
		pos_str = str(self.position)
		rot_str = str(self.rot_axis)
		output = 'Position Array: ' + pos_str +'\n'+'Rotation Axis Array: ' + rot_str
		return output


	def is_self_intersecting(self, position_arr=None, thickness_arr=None):
		"""
		Return True if the given set of points intersect themselves
		"""
		if position_arr == None:
			position_arr = self.get_arm_pos()
		if thickness_arr == None:
			thickness_arr = [.75]*(len(position_arr)-1)

		for i in range(2,len(position_arr)): # do segments hit the ground?
			if position_arr[i][2] < thickness_arr[i-1]:
				return True
			if i < len(thickness_arr) and position_arr[i][2] < thickness_arr[i]:
				return True

		for i in range(1,len(position_arr)-1): # do joints bend too far?
			x0 = Vector(position_arr[i-1])
			x1 = Vector(position_arr[i])
			x2 = Vector(position_arr[i+1])
			v,u = x0-x1, x2-x1
			if v*u > 0:
				return True

		for i in range(0,len(position_arr)-3): # do segments hit each other?
			for j in range(i+2, len(position_arr)-1):
				x0, x1  = Vector(position_arr[i]), Vector(position_arr[i+1])
				if x0 == x1: # define first arm
					break
				y0, y1 = Vector(position_arr[j]), Vector(position_arr[j+1])
				if y0 == y1: # define second arm
					continue
				if dist(x0, x1, y0, y1) <= thickness_arr[i]+thickness_arr[j]:
					return True

		return False


def dist(x0, x1, y0, y1):
	"""
	Distance between line segments x0->x1 and y0->y1
	"""
	SMALL_NUM = 0.01
	u = x1 - x0
	v = y1 - y0
	w = x0 - y0
	a,b,c,d,e = u*u, u*v, v*v, u*w, v*w
	D = a*c - b*b
	sD, tD = D, D

	if D <= SMALL_NUM:
		sN = 0
		sD = 1
		tN = e
		tD = c
	else:
		sN = (b*e - c*d)
		tN = (a*e - b*d)
		if sN < 0:
			sN = 0
			tN = e
			tD = c
		elif sN > sD:
			sN = sD
			tN = e + b
			tD = c

	if tN < 0:
		tN = 0
		if -d < 0:
			sN = 0
		elif -d > a:
			sN = sD
		else:
			sN = -d
			sD = a
	elif tN > tD:
		tN = tD
		if -d + b < 0:
			sN = 0
		elif -d + b > a:
			sN = sD
		else:
			sN = -d + b
			sD = a

	sc = 0 if abs(sN) <= SMALL_NUM else sN/sD
	tc = 0 if abs(tN) <= SMALL_NUM else tN/tD
	return abs(w + u*sc - v*tc)


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



def gen_cylinder(p1=[2,0,6], p2=[6,0,6], r=.75, resolution=15):
		#ax = Axes3D(fig, azim=30, elev=30)
		p1 = np.array(p1)
		p2 = np.array(p2)
		diff = p2-p1
		
		diff_xy = np.array([diff[0], diff[1]])

		if diff_xy[1] < 0:
			azimuth = -math.acos(np.inner(diff_xy/LA.norm(diff_xy), np.array([1,0])))
		else:
			azimuth = math.acos(np.inner(diff_xy/LA.norm(diff_xy), np.array([1,0])))
		elevation = math.acos(np.inner(diff/LA.norm(diff), np.array([0,0,1])))
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
				rot_matrix = rotation_matrix([0,0,1], azimuth) # This is determined by the value of the various sliders
				rot_axis = np.dot(rot_matrix, np.array([0,1,0]))
				rot_matrix_f = rotation_matrix(rot_axis, elevation)
				point = np.dot(rot_matrix_f, point)
				x[i][j] = point[0]+p1[0]
				y[i][j] = point[1]+p1[1]
				z[i][j] = point[2]+p1[2]
		
		return x, y, z


def gen_sphere(p1, r=.75, resolution=50):
	phi, lam = np.meshgrid(np.linspace(-math.pi/2, math.pi/2, resolution), np.linspace(0, 2*math.pi, resolution))
	return r*np.cos(phi)*np.cos(lam)+p1[0], r*np.cos(phi)*np.sin(lam)+p1[1], r*np.sin(phi)+p1[2]


if __name__ == "__main__":

	arm = Arm()
	arm.add_joint([0,1,0], [0, 0, 0])
	arm.add_joint([0,1,0], [2,0,6])
	arm.add_joint([0,1,0], [6,0,6])
	arm.add_joint([0,1,0], [10,0,8])
	arm.add_joint([0,1,0], [13,0,8])
	arm.add_joint([1,2,3], [14,0,5])
	arm.plot_arm()
	#arm.parameter_sweep()
	arm.move_to([3,10,3])
	plt.show()
