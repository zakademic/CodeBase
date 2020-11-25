import numpy as np

#__all__ = ['ConvexHull']

def __IsPointInTriangle(p,q,r,s):
    """
    pqr are three points that form a triangle ordered in CCW order
    s is a point that may be inside the triangle
    Input:
    p: np.shape(p) = [2,]
    q: np.shape(q) = [2,]
    r: np.shape(r) = [2,]
    s: np.shape(s) = [2,]

    Output:
    Return True if point s lies in the triangle pqr
    Return False Otherwise
    """
    to_left_of_pq = __PointToLeftOfVector(p,q,s);
    if (to_left_of_pq == False):
        return False;

    to_left_of_qr = __PointToLeftOfVector(q,r,s);
    if (to_left_of_qr == False):
        return False;

    to_left_of_rp = __PointToLeftOfVector(r,p,s);
    if (to_left_of_rp == False):
        return False;

    return True;

def __PointToLeftOfVector(p,q,s):
    """
    Find if point s is to left of vector formed from pq (p is base point)
    Input: point p,q,s
    p: np.shape(p) = [2,]
    q: np.shape(q) = [2,]
    s: np.shape(r) = [2,]

    Output
    Return True if s is to left of pq
    Return False Otherwise
    """
    orientation = __Orientation(p,q,s);
    if (orientation < 0):
        return False
    return True

def __Orientation(p,q,s):
    """
    Find if point s is to left of vector formed from pq (p is base point)
    Input: point p,q,s
    p: np.shape(p) = [2,]
    q: np.shape(q) = [2,]
    s: np.shape(r) = [2,]

    Output
    Return Negative sign is signed area is negative
    Return Positive sign Otherwise
    """
    pq = q-p;
    pq = pq/np.linalg.norm(pq);

    ps = s-p;
    ps = ps/np.linalg.norm(ps);

    orientation = pq[0]*ps[1] - pq[1]*ps[0];
    if (orientation < 0):
        return -1;

    return 1

def __FindLeftMostPoint(points):

    idx_left = 0;
    pt_left = points[0];
    for idx, point in enumerate(points):
        if (point[0] < pt_left[0]):
            pt_left = point;
            idx_left = idx;
    return pt_left, idx_left

# def __Orientation(hull_base_pt,next_hull_pt,test_point):
#     v1 = next_hull_pt-hull_base_pt;
#     v2 = test_point-hull_base_pt;
#     v1 = v1/np.linalg.norm(v1)
#     v2 = v2/np.linalg.norm(v2)
#     orientation = v1[0]*v2[1] - v1[1]*v2[0]
#     return orientation

def ConvexHull(points):
    #Assert 2D points
    assert(np.shape(points)[0] == 2 or np.shape(points)[1] == 2);
    if (np.shape(points)[0] == 2):
        points = np.transpose(points);
    #Number of points
    npoints = np.shape(points)[0];

    #Find Left Most Point
    pt_left, idx_left = __FindLeftMostPoint(points)
    #List of points in ConvexHull
    hull_pts = [];
    hull_idx = [];
    #Have Now Found Left Most point
    hull_pts.append(pt_left);
    hull_idx.append(idx_left);

    #Hull Base Point
    hull_base_pt = np.copy(pt_left);
    hull_base_idx = idx_left;

    #Loop over all other points
    next_hull_idx = 0;
    # if (idx_left == 0):
    #     next_hull_idx = 1;
    next_hull_pt = np.copy(points[next_hull_idx])
    #print("hull_base_idx: ", hull_base_idx)
    conv = 0;
    count = 0;
    while (conv == 0):
        count += 1;
        next_hull_idx = 0;
        next_hull_pt = points[0];
        for i,point in enumerate(points):
            #Check Orientation
            cross_product = __Orientation(hull_base_pt,next_hull_pt,point)
            #print("cross_product", cross_product)
            if (cross_product < 0):
                next_hull_pt = np.copy(point);
                next_hull_idx = i;
        hull_base_pt = np.copy(next_hull_pt);
        hull_base_idx = next_hull_idx;

        #Append to hull
        hull_pts.append(next_hull_pt)
        hull_idx.append(next_hull_idx)

        # next_hull_idx = idx_left
        # next_hull_pt = np.copy(pt_left)
        #print("next_hull_idx", next_hull_idx)
        if (next_hull_idx == idx_left):
            conv = 1;
        # if (count >= 25):
        #     conv = 1;

    return np.array(hull_idx), np.array(hull_pts)






########
