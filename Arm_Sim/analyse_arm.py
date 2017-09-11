def parameter_sweep(self, index=0, iterations=10000):
		random.seed(5)
		X=[]
		Y=[]
		Z=[]
		for i in range(iterations):
			for j in range(0, len(self.position)):
				self.position[j] = random.uniform(-self.max_angle, self.max_angle)
			point = self.get_cartesian()[-1]
			if not self.is_self_intersecting(position_arr=pose):
				X.append(point[0])
				Y.append(point[1])
				Z.append(point[2])
		x = np.array(X)
		y = np.array(Y)	
		z = np.array(Z)

		xyz = np.vstack([np.array(X), np.array(Y), np.array(Z)])
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