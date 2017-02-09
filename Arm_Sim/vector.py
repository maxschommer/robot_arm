#vector.py
import math

class Vector(object):
	def __init__(self, lst):
		self.x = lst[0]
		self.y = lst[1]
		self.z = lst[2]


	def __eq__(self, other):
		return self.x==other.x and self.y==other.y and self.z==other.z

	def __add__(self, other):
		return Vector([self.x+other.x, self.y+other.y, self.z+other.z])

	def __sub__(self, other):
		return self + (-other)

	def __neg__(self):
		return Vector([-self.x, -self.y, -self.z])

	def __mul__(v, u):
		if isinstance(u, Vector):
			return v.x*u.x + v.y*u.y + v.z*u.z
		else:
			return Vector([v.x*u, v.y*u, v.z*u])

	def __div__(v, c):
		return Vector([v.x/c, v.y/c, v.z/c])

	def __abs__(self):
		return math.sqrt(self.x**2 + self.y**2 + self.z**2)

	def cross(v, u):
		return Vector([v.y*u.z - v.z*u.y,
					   v.z*u.x - v.x*u.z,
					   v.x*u.y - v.y*u.x])

	def hat(self):
		return self / abs(self)

	def __str__(self):
		return "<"+str([self.x, self.y, self.z])[1:-2]+">"