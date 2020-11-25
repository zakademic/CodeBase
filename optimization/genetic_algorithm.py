import numpy as np

class GeneticAlgorithm():
    def __init__(self,cost_function,x0,lb,ub,keep_parents = True,*args,**kwargs):
        self.cost_func = cost_function;
        self.init = np.copy(x0);
        self.num_params = np.shape(x0)[0]
        self.lo_bounds = np.copy(lb);
        self.up_bounds = np.copy(ub);
        assert(self.num_params == np.shape(self.lo_bounds)[0])
        assert(self.num_params == np.shape(self.up_bounds)[0])
        self.keep_parents = keep_parents;
        self.max_iter = 1000;
        self.func_tol = 1e-14;
        self.num_strings = 100;
        self.num_parents = 10;
        self.num_offspring_per_couple = 2;
        self.num_iter = 0;
        self.strings = np.array([]);
        self.min_string = np.array([])
        self.f_vals = np.array([]);
        self.min_cost = 1e9;
        allowed_keys = {'num_parents','num_children_per_parent','num_strings', \
        'beta','gamma','delta','max_iter','func_tol'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    # @property
    # def min_cost():
    #     return self.min_cost;
    def setup_indices(self):
        num_parent_pairs = int(self.num_parents/2)
        num_offspring_per_couple = 2;
        keep_parents = self.keep_parents;
        if (keep_parents):
            num_mutants = int(self.num_strings - self.num_parents*(self.num_offspring_per_couple))
            mutant_index = int(self.num_parents*(self.num_offspring_per_couple))
        else:
            num_mutants = int(self.num_strings - (self.num_parents/2)*self.num_offspring_per_couple);
            mutant_index = int((self.num_parents/2)*self.num_offspring_per_couple);

        return num_parent_pairs, num_mutants, mutant_index

    def mate(self,string1,string2):
        num_vars = np.shape(string1)[0];
        th = np.random.uniform(0,1,num_vars)
        child = (string1-string2)*th + string2;
        return child;

    def optimize(self):
        num_parent_pairs, num_mutants, mutant_index = self.setup_indices();
        conv = 0;
        num_iter = 0;
        min_val = 1e5;

        #Parameter Strings
        strings = np.zeros((self.num_strings,self.num_params))
        param_lo_bound = self.lo_bounds;
        param_up_bound = self.up_bounds;

        while (conv == 0):
            num_iter += 1;
            if (num_iter >= self.max_iter or min_val <= self.func_tol):
                self.min_cost = min_val;
                self.num_iter = num_iter;
                self.strings = np.copy(strings);
                self.f_vals = np.copy(f)
                self.min_string = np.copy(strings[0])
                conv = 1;

            #Create Mutations
            if (num_iter == 1):
                #Generate all strings
                theta = np.random.uniform(0,1,(self.num_strings,self.num_params));
                strings = (param_up_bound-param_lo_bound)*theta + param_lo_bound;
            else:
                if (self.keep_parents):
                #Generate only mutant strings
                    theta = np.random.uniform(0,1,(num_mutants,self.num_params))
                    strings[mutant_index:,:] = (param_up_bound-param_lo_bound)*theta + param_lo_bound;
                else:
                    #Generate mutant strings and replace parents
                    theta = np.random.uniform(0,1,(num_mutants,self.num_params))
                    #Move children to parent position
                    strings[:mutant_index,:] = strings[mutant_index+1:int(self.num_parents*self.num_offspring_per_couple),:]
                    strings[mutant_index:,:] = (param_up_bound-param_lo_bound)*theta + param_lo_bound;

            #Evaluate
            f = [];
            for i in range(0,self.num_strings):
                f.append(np.abs(self.cost_func(strings[i])))

            #Sort
            f = np.array(f);
            idx = np.argsort(f);
            f = f[idx];
            strings = strings[idx];

            #Check Termination Criteria
            min_val = f[0];
            # if (np.abs(min_val) <= TOL_FUNC):
            #     conv = 1;

            if (num_iter >= self.max_iter or np.abs(min_val) <= self.func_tol):
                self.min_cost = min_val;
                self.num_iter = num_iter;
                self.strings = np.copy(strings);
                self.f_vals = np.copy(f)
                self.min_string = np.copy(strings[0])
                conv = 1;

            #Mate
            for i in range(0,num_parent_pairs):
                idx_1 = int(2*i);
                idx_2 = idx_1 + 1;

                for j in range(0,self.num_offspring_per_couple):
                    idx_child = idx_1 + i + self.num_parents + j;

                    #Mate
                    child_string = self.mate(strings[idx_1],strings[idx_2])
                    #Update
                    strings[idx_child] = np.copy(child_string)














        #
