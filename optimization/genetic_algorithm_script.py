import numpy as np
import functions as func


def mate(string1,string2):
    num_vars = np.shape(string1)[0];
    th = np.random.uniform(0,1,num_vars)
    child = (string1-string2)*th + string2;
    return child;

ndim = 2;
num_strings = 100;
num_parents = 10;
num_parent_pairs = int(num_parents/2)
num_offspring_per_couple = 2;
keep_parents = 1;
if (keep_parents):
    num_mutants = int(100 - num_parents*(num_offspring_per_couple))
    mutant_index = int(num_parents*(num_offspring_per_couple))
else:
    num_mutants = int(num_strings - (num_parents/2)*num_offspring_per_couple);
    mutant_index = int((num_parents/2)*num_offspring_per_couple);

#Termination Criteria
MAX_ITER = 10000;
TOL_FUNC = 1e-14;
conv = 0;
num_iter = 0;
min_val = 1e5;

#Optimization Function
opt_func = func.eggholder_func;

#Parameter Bounds
param_lo_bound = np.array([-10,-10])
param_up_bound = np.array([10,10])

#Parameter Strings
strings = np.zeros((num_strings,ndim))

while (conv == 0):
    num_iter += 1;
    if (num_iter >= MAX_ITER):
        conv = 1;
    if (min_val <= TOL_FUNC):
        conv = 1;

    #Create Mutations
    if (num_iter == 1):
        #Generate all strings
        theta = np.random.uniform(0,1,(num_strings,ndim));
        strings = (param_up_bound-param_lo_bound)*theta + param_lo_bound;
    else:
        if (keep_parents):
        #Generate only mutant strings
            theta = np.random.uniform(0,1,(num_mutants,ndim))
            strings[mutant_index:,:] = (param_up_bound-param_lo_bound)*theta + param_lo_bound;
        else:
            #Generate mutant strings and replace parents
            theta = np.random.uniform(0,1,(num_mutants,ndim))
            #Move children to parent position
            strings[:mutant_index,:] = strings[mutant_index+1:int(num_parents*num_offspring_per_couple),:]
            strings[mutant_index:,:] = (param_up_bound-param_lo_bound)*theta + param_lo_bound;

    #Evaluate
    f = [];
    for i in range(0,num_strings):
        f.append(np.abs(opt_func(strings[i])))

    #Sort
    f = np.array(f);
    idx = np.argsort(f);
    f = f[idx];
    strings = strings[idx];

    #Check Termination Criteria
    min_val = f[0];
    if (np.abs(min_val) <= TOL_FUNC):
        conv = 1;

    if (num_iter >= MAX_ITER):
        conv = 1;

    #Mate
    for i in range(0,num_parent_pairs):
        idx_1 = int(2*i);
        idx_2 = idx_1 + 1;

        for j in range(0,num_offspring_per_couple):
            idx_child = idx_1 + i + num_parents + j;

            #Mate
            child_string = mate(strings[idx_1],strings[idx_2])
            #Update
            strings[idx_child] = np.copy(child_string)





















    #
