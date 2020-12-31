import numpy as np 
import cv2 as cv 
import pandas as pd 

import copy 
import glob 
import os 
import argparse 

"""
Resources: 
github 
learn opencv 
microsoft paper
robotics presentation 
stanford 231A 
burger paper 
"""

parameter_dict = {"K_00": None, "K_01": None, "K_02": None, \
    "K_11": None,"K_12": None, "K_22": None, \
    "R_00": None, "R_01": None, "R_02": None, \
    "R_10": None, "R_11": None, "R_12": None, \
    "R_20": None, "R_21": None, "R_22": None, \
    "T_0": None, "T_1": None, "T_2": None}

def solve_homogenous_linear_system(A):

    u, s, vh = np.linalg.svd(A) 
    x = vh[np.argmin(s)]

    return x 

def create_A_matrix(object_points,image_points):

    num_correspondences = len(object_points)
    A = np.zeros((2*num_correspondences,9),dtype=np.float64)

    for i in range(0,num_correspondences): 
        X = object_points[i,0]
        Y = object_points[i,1]
        u = image_points[i,0]
        v = image_points[i,1]

        row_1 = np.array([-X,-Y,-1.0,0,0,0,u*X,u*Y,u])
        row_2 = np.array([0,0,0,-X,-Y,-1.0,v*X,v*Y,v])

        A[(2*i)] = row_1
        A[(2*i)+1] = row_2 


    return A

def find_homography(A):

    h = solve_homogenous_linear_system(A) 
    H = np.reshape(h,(3,3))
    norm = H[2,2]
    H = H[:,:]/norm

    return H 

def create_v_submatrix(H,i,j):
    Hi = H[:,i]
    Hj = H[:,j]
    
    v = np.zeros(6)
    v[0] = Hi[0]*Hj[0]
    v[1] = Hi[0]*Hj[1] + Hi[1]*Hj[0]
    v[2] = Hi[1]*Hj[1]
    v[3] = Hi[2]*Hj[0] + Hi[0]*Hj[2]
    v[4] = Hi[2]*Hj[1] + Hi[1]*Hj[2]
    v[5] = Hi[2]*Hj[2]
    
    return v 

def create_V_matrix(H):
    v_12 = create_v_submatrix(H,0,1)
    v_11 = create_v_submatrix(H,0,0)
    v_22 = create_v_submatrix(H,1,1)
    
    V = np.vstack([v_12,v_11-v_22])
    
    return V 

def find_b_vector(V):
    b = solve_homogenous_linear_system(V) 
    return b 

def create_B_matrix_from_vector(b):
    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[0,2] = b[3]
    B[1,1] = b[2]
    B[1,2] = b[4]
    B[2,2] = b[5]
    
    B[1,0] = B[0,1]
    B[2,0] = B[0,2]
    B[2,1] = B[1,2]
    
    return B 

def get_K_from_B(B):

    L = np.linalg.cholesky(B) 
    s = L[2,2]
    K = s * np.linalg.inv(L.T)

    return K 

def get_K_from_b_zhang(b):

    vc = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + vc*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((l/b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) *(beta/l))
    uc = (gamma*vc/beta) - (b[3]*(alpha**2)/l)

    K = np.array([
            [alpha, gamma, uc],
            [0, beta, vc],
            [0, 0, 1.0],
        ])

    return K 

def get_extrinsics(K,H):
    Kinv = np.linalg.inv(K)
    lmbda = 1/np.linalg.norm(np.dot(Kinv,H[:,0]))

    r0 = lmbda * np.dot(Kinv,H[:,0])
    r1 = lmbda * np.dot(Kinv,H[:,1])
    t = lmbda * np.dot(Kinv,H[:,2])

    r2 = np.cross(r0,r1)

    Rhat = np.zeros((3,3))
    Rhat[:,0] = r0 
    Rhat[:,1] = r1 
    Rhat[:,2] = r2 

    u,s,vh = np.linalg.svd(Rhat)
    R = np.dot(u,vh)

    #Ensure rotation matrix is orthogonal 
    assert(np.abs(np.dot(R[:,0],R[:,1])) <= 1e-10)
    assert(np.abs(np.dot(R[:,0],R[:,2])) <= 1e-10)
    assert(np.abs(np.dot(R[:,2],R[:,1])) <= 1e-10)

    return R, t

def read_correspondeces_from_folder(folder_name):

    files = glob.glob(os.path.join(folder_name,"*.csv"))

    object_points_list = []
    image_points_list = []
    for file in files: 
        df = pd.read_csv(file) 
        object_points = df[["World X","World Y"]].values 
        image_points = df[["Image X","Image Y"]].values

        object_points_list.append(object_points)
        image_points_list.append(image_points)

    return object_points_list, image_points_list

def serialize_calibration_parameters(K,R_list,T_list,output_folder):

    param_dict_list = []
    for (R,T) in zip(R_list,T_list):

        rvec = R.flatten() 
        tvec = T.flatten() 

        param_dict = copy.deepcopy(parameter_dict)
        param_dict["K_00"] = K[0,0]
        param_dict["K_01"] = K[0,1]
        param_dict["K_02"] = K[0,2]
        param_dict["K_11"] = K[1,1]
        param_dict["K_12"] = K[1,2]
        param_dict["K_22"] = K[2,2]

        param_dict["R_00"] = R[0,0]
        param_dict["R_01"] = R[0,1]
        param_dict["R_02"] = R[0,2]

        param_dict["R_10"] = R[1,0]
        param_dict["R_11"] = R[1,1]
        param_dict["R_12"] = R[1,2]

        param_dict["R_20"] = R[2,0]
        param_dict["R_21"] = R[2,1]
        param_dict["R_22"] = R[2,2]

        param_dict["T_0"] = T[0]
        param_dict["T_1"] = T[1]
        param_dict["T_2"] = T[2]

        param_dict_list.append(param_dict)

    #Combine all dicts 
    combined_dict = {}
    for k in parameter_dict.keys():
        combined_dict[k] = list(combined_dict[k] for combined_dict in param_dict_list)

    df = pd.DataFrame(combined_dict) 
    num_views = len(R_list)
    view_array = np.arange(num_views)
    df.insert(0,"View", view_array)
    df.to_csv(os.path.join(output_folder,"calibration_parameters.csv"),index=False)
    
    return 0 

def rms_error(H_list, object_points_list, image_points_list): 
    
    rms = 0
    num_samples = 0
    for k,H in enumerate(H_list): 
        object_points = object_points_list[k]
        image_points = image_points_list[k]
        
        num_samples += len(object_points)
        
        for (object_point, image_point) in zip(object_points,image_points): 
            
            homogenous_object_point = np.array([object_point[0],object_point[1],1.0])
            predicted_image_point = np.dot(H,homogenous_object_point)
            #Only know the predicted image points up to a scale, divide by last value
            predicted_image_point = predicted_image_point/predicted_image_point[2]
            predicted_image_point = predicted_image_point[:2]
            
            error = np.linalg.norm(predicted_image_point-image_point)
            
            rms += error 
            
    #After all views and all correspondences calculated, find rms error 
    rms = np.sqrt(rms/num_samples)
    
    return rms 

def main(correspondence_folder,output_folder=None,reprojection_error=None):

    object_points_list, image_points_list = read_correspondeces_from_folder(correspondence_folder)

    A_list = [] 
    for (object_points,image_points) in zip(object_points_list,image_points_list): 
        A = create_A_matrix(object_points,image_points)
        A_list.append(A) 

    H_list = [] 
    for A in A_list: 
        H = find_homography(A) 
        H_list.append(H) 

    #Find Intrinsicss
    V_list = []
    for H in H_list: 
        V = create_V_matrix(H)
        V_list.append(V)
    V_all_views = np.vstack(V_list)
    b = find_b_vector(V_all_views) 
    #Two methods to find Intrinsic Parameters
    Kzhang = get_K_from_b_zhang(b) 
    B = create_B_matrix_from_vector(b)
    Kcholesky = get_K_from_B(B) 

    R_list = [] 
    T_list = [] 
    for H in H_list: 
        R, T = get_extrinsics(Kcholesky,H)
        R_list.append(R)
        T_list.append(T)

    if output_folder is None: 
        output_folder = os.path.join(correspondence_folder,"output")
        os.makedirs(output_folder,exist_ok=True)

    serialize_calibration_parameters(Kcholesky,R_list,T_list,output_folder)

    print("Camera Intrinsic Matrix")
    print(Kcholesky)

    if reprojection_error is not None:
        rms = rms_error(H_list, object_points_list, image_points_list)
        print("RMS Error: ", rms, "pixels.")

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Camera Calibration")

    parser.add_argument("--correspondence_folder",type=str,help="Folder with image correspondence files.")
    parser.add_argument("--output_folder",type=str,default=None,help="Folder where output is written.")
    parser.add_argument("--reprojection_error",action="store_true",default=None,help="Calculate reprojection error?")
    args = parser.parse_args() 
    exit(main(args.correspondence_folder,args.output_folder,args.reprojection_error))