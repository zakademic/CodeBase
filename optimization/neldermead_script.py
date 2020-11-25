import numpy as np
import functions as func
import matplotlib.pyplot as plt
import time

#Params
alpha = 1;
beta = 0.5;
gamma = 2;
delta = 0.5;
TOL_FUNC = 1e-14;
MAX_ITER = 10000;
num_iter = 0;
opt_func = func.parab_func;

#Initial Guess
factor = 10;
x0 = np.array([0.2,0.2])
x1 = np.array([0.3,0.2])
x2 = np.array([0.2,0.25])
points = np.array([x0,x1,x2])*factor
num_points = np.shape(points)[0]

#Plotting
plot_simplex = 1;
axes = plt.gca()
ax_lim = 5;
axes.set_xlim(-ax_lim, ax_lim)
axes.set_ylim(-ax_lim, ax_lim)
xs = points[:,0]; xs = np.append(xs,points[0,0])
ys = points[:,1]; ys = np.append(ys,points[0,1])
line, = axes.plot(xs, ys, 'r-')
xfunc = np.linspace(-ax_lim, ax_lim);
yfunc = np.linspace(-ax_lim, ax_lim);
zfunc = np.zeros((np.shape(yfunc)[0],np.shape(xfunc)[0]))
for i,x in enumerate(xfunc):
    for j,y in enumerate(yfunc):
        zfunc[j,i] = opt_func([x,y])
plt.contourf(xfunc,yfunc,zfunc);
plt.axis('equal')
#line, = axes.plot();

conv = 0;
while (conv == 0):
    num_iter += 1;
    #Evaluate Points
    f = [];
    for point in points:
        f.append(opt_func(point))
    #Sort Points
    idx = np.argsort(f);
    #Best (l), Second Worst (s), Worst (h);
    f = np.array(f);
    f = f[idx];
    points = points[idx]
    fl = f[0]; xl = points[0]
    fs = f[-2]; xs = points[-2]
    fh = f[-1]; xh = points[-1]

    if (plot_simplex):
        xplot = np.array([xl[0],xs[0],xh[0],xl[0]]);
        yplot = np.array([xl[1],xs[1],xh[1],xl[1]]);
        line.set_xdata(xplot)
        line.set_ydata(yplot)
        plt.draw()
        plt.pause(0.5)
        time.sleep(0.1)

    #Check Termination Criteria
    if (fl <= TOL_FUNC):
        conv = 1;
    if (num_iter >= MAX_ITER):
        conv = 1;

    #Compute Centroid
    centroid = np.zeros(num_points-1)
    for i in range(0,num_points-1):
        centroid += points[i];
    centroid /= (num_points-1);

    #Change Simplex
    #Reflect
    xr = centroid + alpha * (centroid-xh);
    fr = opt_func(xr);

    #Expand
    if (fr < fs):
        if (fr >= fl):
            #xh = xr;
            points[-1] = np.copy(xr);
            #Terminate Iteration
            continue;
        else:
            #Compute Expansion
            xe = centroid + gamma * (xr-centroid)
            fe = opt_func(xe);

            if (fe < fr):
                #xh = xe;
                points[-1] = np.copy(xe);
                #Terminate Iteration
                continue;
            else:
                #xh = xr;
                points[-1] = np.copy(xr);
                #Terminate Iteration
                continue;
            #ADD GREEDY EXPANSION LATER

    #Contraction
    if (fr >= fs):
        if (fr < fh):
            xc = centroid + beta*(xr-centroid);
            fc = opt_func(xc);
            if (fc <= fr):
                #xh = xc;
                points[-1] = np.copy(xc);
                #Terminate Iteration
                continue;
            else:
                #Shrink
                for i in range(1,num_points):
                    points[i] = xl + delta*(points[i]-xl);
                #Terminate Iteration
                continue;
        if (fr >= fh):
            xc = centroid + beta*(xh-centroid);
            fc = opt_func(xc);
            if (fc < fh):
                #xh = xc;
                points[-1] = np.copy(xc);
                #Terminate Iteration
                continue;
            else:
                #Shrink
                for i in range(1,num_points):
                    points[i] = xl + delta*(points[i]-xl);
                #Terminate Iteration
                continue;


plt.show()















#
