import numpy as np

class Point2D:
    def __init__(self,point):
        assert(np.shape(point)[0] == 2)
        self.x = point[0];
        self.y = point[1];

    def __init__(self,x,y):
        self.x = x;
        self.y = y;

class Point3D:
    def __init__(self,point):
        assert(np.shape(np_array)[0] == 3);
        self.x = point[0];
        self.y = point[1];
        self.z = point[2];

    def __init__(self,x,y,z):
        self.x = x;
        self.y = y;
        self.z = z;

class Circle2D:
    def __init__(self,x,y,z,radius):
        self.x0 = x;
        self.y0 = y;
        self.z0 = z;
        self.radius = radius;
        self.center = np.array([x,y,z]);

    def __init__(self,center,radius):
        self.x0 = center[0];
        self.y0 = center[1];
        self.z0 = center[2];
        self.radius = radius;
        self.center = np.copy(center)

class Circle3D:
    def __init__(self,x,y,z,radius,normal):
        self.x0 = x;
        self.y0 = y;
        self.z0 = z;
        self.radius = radius;
        self.center = np.array([x,y,z]);
        self.normal = np.copy(normal);

    def __init__(self,center,radius,normal):
        self.x0 = center[0];
        self.y0 = center[1];
        self.z0 = center[2];
        self.radius = radius;
        self.center = np.copy(center);
        self.normal = np.copy(normal);

class Ray2D:
    def __init__(self,base,direction):
        assert(np.shape(base)[0] == 2);
        assert(np.shape(direction)[0] == 2);
        self.base = np.copy(base);
        self.direction = np.copy(direction);

class Ray3D:
    def __init__(self,base,direction):
        assert(np.shape(base)[0] == 3);
        assert(np.shape(direction)[0] == 3);
        self.base = np.copy(base);
        self.direction = np.copy(direction);
