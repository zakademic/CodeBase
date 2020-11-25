import numpy as np
import geometry.ConvexHull as ch

def PointInTriangleTest():
    """
    Form a triangle and place point inside, make sure returns True
    Then place point outside and make sure returns false
    """

    p = np.array([0.0,0.0])
    q = np.array([2.0,0.0])
    r = np.array([0.0,2.0])

    s_in = np.array([0.1,0.1])
    s_out = np.array([2.0,2.0])

    is_s_in_inside_triangle = ch.__IsPointInTriangle(p,q,r,s_in);
    is_s_out_inside_triangle = ch.__IsPointInTriangle(p,q,r,s_out);

    print(is_s_in_inside_triangle);
    print(is_s_out_inside_triangle)

    test_passed = (is_s_in_inside_triangle == True) and (is_s_out_inside_triangle == False)
    return test_passed



#Run Tests

point_in_triangle_test = PointInTriangleTest();
print(point_in_triangle_test)
p = np.array([0.0,0.0])
q = np.array([2.0,0.0])
r = np.array([0.0,2.0])

s_in = np.array([0.1,0.1])
s_out = np.array([2.0,2.0])
