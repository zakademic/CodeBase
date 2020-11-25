import numpy as np
import matplotlib.pyplot as plt
import geometry.ConvexHull as ch
import datatypes.datatypes as dt

point1 = dt.Point2D(2,4)

#create many random points
npts = 100;
xmin = -10.0; xmax = 10.0;
ymin = -10.0; ymax = 10.0;

np.random.seed(0)
points = np.random.rand(npts,2);
points[:,0] = points[:,0]*(xmax-xmin) + xmin
points[:,1] = points[:,1]*(ymax-ymin) + ymin

#Plot points
plt.figure();
plt.scatter(points[:,0],points[:,1])

cvx_hull_idx, cvx_hull_pts = ch.ConvexHull(points)
plt.plot(cvx_hull_pts[:,0],cvx_hull_pts[:,1],'go:')
plt.show()
plt.close('all')
