import numpy as np
import math
import random

from vector import Vector


class Arm:
	"""This holds all of the positions of all of the joints, and the axis of roation of all of the segments""" 
	def __init__(self, lengths=None, direction=0, position=None, max_angle=math.pi/2, base_height=1, arm_size=1, joint_size=1.5, manip_size=2):
		self.lengths = [] if lengths is None else lengths #the segments, each represented by a length
		self.direction = 0 #the position of the base rotation joint
		self.position = [0]*len(self.lengths) if position is None else position #the position of each elbow joint (0=straight)
		self.max_angle = max_angle #the max angle a joint can bend
		self.base_height = base_height #the hegiht of the basemost elbow joint
		self.arm_size = arm_size #the thickness of the arm
		self.joint_size = joint_size #the diameter of each joint
		self.manip_size = manip_size #the diameter of the manipulator


	def get_cartesian(self):
		""" Returns a list of coordinate vectors with each joint in cartesian coordinates """
		xfac = math.cos(self.direction)
		yfac = math.sin(self.direction)
		t, r, z = 0, 0, self.base_height
		locations = [[r,r,z]]
		for posn, leng in zip(self.lengths, self.position):
			t += posn
			r += leng*math.sin(t)
			z += leng*math.cos(t)
			locations.append([r*xfac, r*yfac, z])
		return locations


	def is_self_intersecting(self, position=None):
		""" Return True if the given set of points intersect themselves """
		if position is not None:
			self.position = position
		locations = np.array(self.get_cartesian())

		for joint_pos in locations: # do segments hit the ground?
			if joint_pos[2] <= self.joint_size/2:
				return True

		for angle in self.position: #do joints bend too far?
			if abs(angle) >= self.max_angle:
				return True

		for i in range(0,len(position_arr)-3): # do segments hit each other?
			for j in range(i+2, len(position_arr)-1):
				ra0, ra1  = Vector(position_arr[i]), Vector(position_arr[i+1]) # define first arm
				rb0, rb1 = Vector(position_arr[j]), Vector(position_arr[j+1]) # define second arm
				if dist(ra0, rb0) <= self.joint_size or dist(ra0, rb1) <= self.joint_size or \
						dist(ra1, rb0) <= self.joint_size or dist(ra1, rb1) <= self.joint_size:
					return True
				if dist(ra0, ra0, rb0, rb1) <= (self.arm_size+self.joint_size)/2 or \
						dist(ra1, ra1, rb0, rb1) <= (self.arm_size+self.joint_size)/2 or \
						dist(ra0, ra1, rb0, rb0) <= (self.arm_size+self.joint_size)/2 or \
						dist(ra0, ra1, rb1, rb1) <= (self.arm_size+self.joint_size)/2:
					return True
				if dist(ra0, ra1, rb0, rb1) <= self.arm_size:
					return True

		return False


def dist(x, y):
	""" Distance between points x and y """
	return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)


def dist(x0, x1, y0, y1):
	""" Distance between line segments x0->x1 and y0->y1 """
	SMALL_NUM = 0.01 #This is a really small number
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
