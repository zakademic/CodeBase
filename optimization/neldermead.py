import numpy as np

class NelderMead:
    def __init__(self,cost_function,x0,*args,**kwargs):
        self.func = cost_function;
        self.points = x0;
        self.xl = x0[0];
        self.fl = self.func(self.xl);
        self.num_iter = 0;
        self.num_points = np.shape(x0)[0];
        self.alpha = 1;
        self.beta = 0.5;
        self.gamma = 2;
        self.delta = 0.5;
        self.max_iter = 1000;
        self.func_tol = 1e-16;
        allowed_keys = {'alpha','beta','gamma','delta','max_iter','func_tol'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    def optimize(self):
        num_iter = 0;
        conv = 0;
        while (conv == 0):
            num_iter += 1;
            #Evaluate Points
            f = [];
            for point in self.points:
                f.append(self.func(point))
            #Sort Points
            idx = np.argsort(f);
            #Best (l), Second Worst (s), Worst (h);
            f = np.array(f);
            f = f[idx];
            self.points = self.points[idx]
            fl = f[0]; xl = self.points[0]
            fs = f[-2]; xs = self.points[-2]
            fh = f[-1]; xh = self.points[-1]

            #Check Termination Criteria
            if (fl <= self.func_tol):
                conv = 1;
            if (num_iter >= self.max_iter):
                conv = 1;

            #Compute Centroid
            centroid = np.zeros(self.num_points-1)
            for i in range(0,self.num_points-1):
                centroid += self.points[i];
            centroid /= (self.num_points-1);

            #Transform Simplex
            #Reflect
            xr,fr = self.reflect(xh,centroid);

            #Expand
            if (fr < fs):
                if (fr >= fl):
                    #xh = xr;
                    self.points[-1] = np.copy(xr);
                    #Terminate Iteration
                    continue;
                else:
                    #Compute Expansion
                    xe,fe = self.expand(xr,centroid);

                    if (fe < fr):
                        #xh = xe;
                        self.points[-1] = np.copy(xe);
                        #Terminate Iteration
                        continue;
                    else:
                        #xh = xr;
                        self.points[-1] = np.copy(xr);
                        #Terminate Iteration
                        continue;
                    #ADD GREEDY EXPANSION LATER

            #Contraction
            if (fr >= fs):
                if (fr < fh):
                    #Outside Contraction
                    xc,fc = self.outside_contract(xr,centroid);
                    if (fc <= fr):
                        #xh = xc;
                        self.points[-1] = np.copy(xc);
                        #Terminate Iteration
                        continue;
                    else:
                        #Shrink
                        # for i in range(1,self.num_points):
                        #     self.points[i] = xl + self.delta*(points[i]-xl);
                        self.shrink(xl)
                        #Terminate Iteration
                        continue;
                if (fr >= fh):
                    #Inside Contraction
                    xc,fc = self.inside_contract(xh,centroid);
                    if (fc < fh):
                        #xh = xc;
                        self.points[-1] = np.copy(xc);
                        #Terminate Iteration
                        continue;
                    else:
                        #Shrink
                        self.shrink(xl)
                        #Terminate Iteration
                        continue;


        #Update values
        self.xl = np.copy(xl);
        self.fl = fl;
        self.num_iter = num_iter;

        return None


    def reflect(self,xh,centroid):
        xr = centroid + self.alpha * (centroid-xh);
        fr = self.func(xr);
        return xr,fr
    def expand(self,xr,centroid):
        xe = centroid + self.gamma * (xr-centroid)
        fe = self.func(xe);
        return xe,fe
    def outside_contract(self,xr,centroid):
        xc = centroid + self.beta*(xr-centroid);
        fc = self.func(xc);
        return xc,fc
    def inside_contract(self,xh,centroid):
        xc = centroid + self.beta*(xh-centroid);
        fc = self.func(xc);
        return xc,fc
    def shrink(self,xl):
        #Shrink
        for i in range(1,self.num_points):
            self.points[i] = xl + self.delta*(points[i]-xl);




































#
