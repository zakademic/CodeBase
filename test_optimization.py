import numpy as np
import optimization.neldermead as nm
import optimization.genetic_algorithm as ga
import optimization.functions as func

def test_neldermead():
    #Initial Guess
    x0 = np.array([0.2,0.2])
    x1 = np.array([0.3,0.2])
    x2 = np.array([0.2,0.25])

    points = np.array([x0,x1,x2])
    opt_func = func.parab_func;

    #Create Nelder Mead Instance
    optx = nm.NelderMead(opt_func,points);
    optx.optimize();

    return optx;

optx = test_neldermead();

def test_ga():

    #Choose Optimization Function
    opt_func = func.parab_func;

    #Choose initial point, lower bounds, upper bounds
    x0 = np.array([4.4,2.2])
    lb = np.array([-100,-100])
    ub = np.array([100,100])

    optx = ga.GeneticAlgorithm(opt_func,x0,lb,ub)
    optx.optimize();
    return optx;

ga_opt = test_ga();
