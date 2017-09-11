#interactive_arm.py

def plot_arm(self, limits=[-10,10,-10,10,0,10], static_plot=True):
	color = 'r' if self.is_self_intersecting(self.rest_position) else 'b'

	plt.subplots_adjust(bottom=(0.05*len(self.rest_position)))

	if static_plot == True:
		#self.ax.plot(*np.array(self.lines).T)

		for i in range(len(self.rest_position)-1):
			if not np.array_equal(self.rest_position[i],  self.rest_position[i+1]):
				x, y, z = gen_cylinder(self.rest_position[i], self.rest_position[i+1])
				self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color)
				x, y, z = gen_sphere(self.rest_position[i+1])
				self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color='k')

	else:
		if len(self.history) > 0:
			for i in range(len(self.rest_position)):
				self.ax.plot(*np.array(self.history)[:,i,:].T)

		#self.ax.plot(*np.array(self.lines).T)
   		current_position = self.get_arm_pos()
		for i in range(len(current_position)-1):
			if not np.array_equal(current_position[i],  current_position[i+1]):
				x, y, z = gen_cylinder(current_position[i], current_position[i+1])
				self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color)
				x, y, z = gen_sphere(current_position[i+1])
				self.ax.plot_surface(x, y, z, rstride=4, cstride=4, color='k')

	for i in range(len(self.rest_position)-1):
		self.s_joints[i].on_changed(self.update)

	self.ax.set_xlim3d(-10, 10)
	self.ax.set_ylim3d(-10, 10)
	self.ax.set_zlim3d(0, 20)
	self.ax.grid(b=False)

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
	self.ax.axis('square')
	self.fig.canvas.draw_idle()
