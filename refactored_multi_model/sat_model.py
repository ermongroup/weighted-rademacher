#Code for estimating #SAT (the number of satisfying assignments).  
#We solve a SAT problem, perturbed to be a partial maximum satisfiability problem.
from __future__ import division

#import subprocess
import commands
import numpy as np
import copy
from decimal import Decimal
from boundZ import lower_bound_Z as gen_lower_bound_Z
from boundZ import upper_bound_Z as gen_upper_bound_Z
from boundZ import calculate_gumbel_slack

import pickle
import time

import matplotlib
matplotlib.use('Agg') #prevent error running remotely
import matplotlib.pyplot as plt

from chunked_perturbations import sample_chunked_perturbations
from chunked_perturbations import get_all_vecs
from chunked_perturbations import calc_chunked_perturbation

#Directory the open-wbo_static executable is in
#Install wbo from http://sat.inesc-id.pt/open-wbo/installation.html
WBO_DIRECTORY = '/atlas/u/jkuck/software/open-wbo'

#Directory with the MaxHS executable
#Download from http://www.maxhs.org/downloads.html
MAX_HS_DIRECTORY = '/atlas/u/jkuck/software/MaxHS-3.0/build/release/bin'

SAT_SOLVER = "MAX_HS"
#SAT_SOLVER = "WBO" #something strange seems to happen using WBO, also seems slower than MAX_HS

class SAT_problem:
    def __init__(self, nbvar, nbclauses, clauses):
        '''
        Represent an unweighted SAT problem  

        Inputs:
        - nbvar: int, number of variables 
        - nbclauses: int, the number of clauses
        - clauses: list of strings, each string represents a clause as a sequence of non-zero
            integer numbers between -nbvar and nbvar and ends with 0. Positive numbers denote 
            the corresponding variables. Negative numbers denote the negations of the 
            corresponding variables. (as in http://www.maxhs.org/docs/wdimacs.html)
        '''
        self.nbvar = nbvar
        self.nbclauses = nbclauses
        self.clauses = clauses

        #may be set to 'gumbel', 'barvinok', or 'chunked_barvinok'
        self.perturbation_type = None

    def gumbel_perturb(self):
        '''
        Turn the SAT problem into a weighted partial MaxSAT, perturbing with Gumbel noise
        '''
        #many Max-SAT solvers require that top be larger than the sum of the weights of the 
        #falsified clauses in an optimal solution (top being the sum of the weights of all 
        #soft clauses plus 1 will always suffice)
        self.perturbation_type = 'gumbel'
        self.top = 99 
        #list of strings, each representing a soft clause. Clauses are expressed as above, 
        #with the addition that the first number of each clause line identifies the weight of the clause
        self.soft_clauses = []

        #gumbel perturbation for each variable taking the value +1
        self.pos1_gumbel_perturbs = []
        #gumbel perturbation for each variable taking the value -1
        self.neg1_gumbel_perturbs = []
        for i in range(self.nbvar):
            cur_p1_pert = np.random.gumbel(loc=-np.euler_gamma, scale=1.0)
            cur_n1_pert = np.random.gumbel(loc=-np.euler_gamma, scale=1.0)
            self.pos1_gumbel_perturbs.append(cur_p1_pert)
            self.neg1_gumbel_perturbs.append(cur_n1_pert)
            #set weight for x_i = 1 to positive gumbel difference, leaving weight for x_i = -1 at zero              
            if cur_p1_pert > cur_n1_pert:
                if SAT_SOLVER == "MAX_HS":
                    clause_weight = cur_p1_pert - cur_n1_pert
                    cur_soft_clause = "%f %d 0\n" % (clause_weight, i+1) #SAT solver expects 1 indexed variables
                elif SAT_SOLVER == "WBO":
                    clause_weight = int(round(100*(cur_p1_pert - cur_n1_pert)))
                    cur_soft_clause = "%d %d 0\n" % (clause_weight, i+1) #SAT solver expects 1 indexed variables
                else:
                    assert(False), ("Invalid SAT_SOLVER")
                self.top += clause_weight
            #set weight for x_i = -1 to positive gumbel difference, leaving weight for x_i = 1 at zero              
            else:
                if SAT_SOLVER == "MAX_HS":
                    clause_weight =  cur_n1_pert - cur_p1_pert
                    cur_soft_clause = "%f -%d 0\n" % (clause_weight, i+1) #SAT solver expects 1 indexed variables
                elif SAT_SOLVER == "WBO":
                    clause_weight =  int(round(100*(cur_n1_pert - cur_p1_pert)))
                    cur_soft_clause = "%d -%d 0\n" % (clause_weight, i+1) #SAT solver expects 1 indexed variables
                else:
                    assert(False), ("Invalid SAT_SOLVER")       
                self.top += clause_weight
            self.soft_clauses.append(cur_soft_clause)
        assert(len(self.soft_clauses) == self.nbvar)

    def barvinok_perturb(self):
        '''
        Turn the SAT problem into a weighted partial MaxSAT, perturbing with {-1,1}^n noise
        sampled uniformly at random (n = nbvar)
        '''
        self.perturbation_type = 'barvinok'

        #many Max-SAT solvers require that top be larger than the sum of the weights of the 
        #falsified clauses in an optimal solution (top being the sum of the weights of all 
        #soft clauses plus 1 will always suffice)
        self.top = 99 

        #list of strings, each representing a soft clause. Clauses are expressed as above, 
        #with the addition that the first number of each clause line identifies the weight of the clause
        self.soft_clauses = []
        #store the random c vector we are perturbing with explicitly
        self.perturbation_c = []
        for i in range(self.nbvar):
            self.top += 2
            c_i = np.random.rand()
            #c_i = -1, set weight for x_i = -1 to two, leaving weight for x_i = 1 at zero
            if c_i < .5: 
                self.perturbation_c.append(-1)
                cur_soft_clause = "2 -%d 0\n" % (i+1) #SAT solver expects 1 indexed variables
            #c_i = 1, set weight for x_i = 1 to two, leaving weight for x_i = -1 at zero                
            else:
                self.perturbation_c.append(1)
                cur_soft_clause = "2 %d 0\n" % (i+1) #SAT solver expects 1 indexed variables
            self.soft_clauses.append(cur_soft_clause)
        assert(len(self.soft_clauses) == self.nbvar)

    def dot_product_perturb_gumbel(self, gumbel_location):
        '''
        Turn the SAT problem into a weighted partial MaxSAT, 
        perturbing to find max_x{<c,x>} such that x is a satisfying solution
        and c is a vector of length n with c_i sampled from the Gumbel distribution
        Inputs:
        - gumbel_location: float, c_i is sampled from the Gumbel distribution with location gumbel_location
        '''
        self.perturbation_type = 'barvinok'

        #many Max-SAT solvers require that top be larger than the sum of the weights of the 
        #falsified clauses in an optimal solution (top being the sum of the weights of all 
        #soft clauses plus 1 will always suffice)
        self.top = 99 

        #list of strings, each representing a soft clause. Clauses are expressed as above, 
        #with the addition that the first number of each clause line identifies the weight of the clause
        self.soft_clauses = []
        #store the random c vector we are perturbing with explicitly
        self.perturbation_c = []
        for i in range(self.nbvar):
            c_i = np.random.gumbel(loc=gumbel_location, scale=1.0)
            self.perturbation_c.append(c_i)
            self.top += 2*np.abs(c_i)
            #c_i < 0, set weight for x_i = -1 to -2*c_i, leaving weight for x_i = 1 at zero
            if c_i < 0: 
                cur_soft_clause = "%f -%d 0\n" % (-2*c_i ,(i+1)) #SAT solver expects 1 indexed variables
            #c_i >= 0, set weight for x_i = 1 to 2*c_i, leaving weight for x_i = -1 at zero                
            else:
                cur_soft_clause = "%f %d 0\n" % (2*c_i, (i+1)) #SAT solver expects 1 indexed variables
            self.soft_clauses.append(cur_soft_clause)
        assert(len(self.soft_clauses) == self.nbvar)


    def barvinok_perturb_chunked(self, chunk_size):
        '''
        Turn the SAT problem into a weighted partial MaxSAT. For n variables, perturb
        groups of chunk_size variables with 2^chunk_size bits of randomness sampled from {-1,1} 
        uniformly at random.  (if n (nbvar) is not divisible by chunk_size, one group will be
        smaller than chunk size and perturbed with 2^{its group size} random bits)
        '''
        self.perturbation_type = 'chunked_barvinok'

        self.chunked_perturbations = sample_chunked_perturbations(n=self.nbvar, m=chunk_size)
        self.chunk_size = chunk_size

        #many Max-SAT solvers require that top be larger than the sum of the weights of the 
        #falsified clauses in an optimal solution (top being the sum of the weights of all 
        #soft clauses plus 1 will always suffice)
        self.top = 99 

        #list of strings, each representing a soft clause. Clauses are expressed as above, 
        #with the addition that the first number of each clause line identifies the weight of the clause
        self.soft_clauses = []

        def calc_dot_product(x, y):
            assert(len(x) == len(y))
            return sum(x_i*y_i for x_i,y_i in zip(x, y))
        
        chunk_size_vecs = get_all_vecs(chunk_size)#get all 2^chunk_size vectors in {-1,1}^chunk_size 
        chunk_idx = 0
        for vec_idx in range(0, self.nbvar, chunk_size):
            cur_chunk_size = min(self.nbvar - vec_idx, chunk_size)
            assert(cur_chunk_size <= chunk_size), (cur_chunk_size, chunk_size)
            if cur_chunk_size == chunk_size:
                cur_chunk_size_vecs = chunk_size_vecs
            else:
                cur_chunk_size_vecs = get_all_vecs(cur_chunk_size)
            #iterate through all 2^cur_chunk_size vectors, creating a soft clause for each with 
            #weight given by its associated perturbation    
            for chunk_vec in cur_chunk_size_vecs: 
                #the perturbation we sampled associated with the current chunk of the vector 
                #taking the values specified by chunk_vec
                cur_chunk_perturbation = self.chunked_perturbations[chunk_idx][chunk_vec]

                #SAT solver needs positive weights, so add a constant to all weights 
                #(actually non-negative may be fine, but making positive to avoid checking)
                positive_cur_chunk_perturbation = cur_chunk_perturbation + chunk_size + 1
                assert(positive_cur_chunk_perturbation >= 0)
                self.top += positive_cur_chunk_perturbation 

                cur_soft_clause = "%d " % positive_cur_chunk_perturbation
                #append variable values
                #variable val = -1 or 1
                #var_idx is the index of the variable within the chunk, 0 to cur_chunk_size-1
                for (var_idx, variable_val) in enumerate(chunk_vec): 
                    assert(variable_val == -1 or variable_val == 1)
                    cur_soft_clause += "%d " % (variable_val*(chunk_idx*chunk_size + var_idx + 1)) #SAT solver expects 1 indexed variables
                cur_soft_clause +=  "0\n"
                self.soft_clauses.append(cur_soft_clause)
            chunk_idx+=1
        assert(chunk_idx == len(self.chunked_perturbations))

def get_delta(perturbed_SAT, max_state):
    '''
    Get the value of max_x <c,x> after finding the max_state x for the SAT problem perturbed with random c
    OR
    the biggest perturbation for chunked perturbation

    Inputs:
    - perturbed_SAT: type SAT_problem, must be a weighted partial MaxSAT problem, perturbed
        with a c vector in {-1,1}^n
    - max_state: list of ints, each entry is either 1 or -1. 

    Outputs:
    - delta: int, max_x <c,x> for x that are satisfying solutions to the original SAT problem

    '''
    if perturbed_SAT.perturbation_type == 'barvinok':
        assert(len(perturbed_SAT.perturbation_c) == len(max_state))
        delta = 0
        for i in range(len(max_state)):
            delta += perturbed_SAT.perturbation_c[i]*max_state[i]
    else:
        assert(perturbed_SAT.perturbation_type == 'chunked_barvinok'), perturbed_SAT.perturbation_type
        delta = calc_chunked_perturbation(perturbed_SAT.chunked_perturbations, tuple(max_state), n=len(max_state), m=perturbed_SAT.chunk_size)
    return delta

def get_gumbel_perturbed_max(perturbed_SAT, max_state):
    '''
    Get the value of max gumbel perturbations for the specified max_state

    Inputs:
    - perturbed_SAT: type SAT_problem, must be a weighted partial MaxSAT problem, perturbed
        with two vectors of gumbel noise, perturbed_SAT.pos1_gumbel_perturbs and perturbed_SAT.neg1_gumbel_perturbs
    - max_state: list of ints, each entry is either 1 or -1. 

    Outputs:
    - gumbel_perturb: float, sum of gumbel perturbations for max_state variable states

    '''
    assert(len(perturbed_SAT.pos1_gumbel_perturbs) == len(perturbed_SAT.neg1_gumbel_perturbs))
    assert(len(perturbed_SAT.pos1_gumbel_perturbs) == len(max_state)), (len(perturbed_SAT.pos1_gumbel_perturbs), len(max_state))
    gumbel_perturb = 0.0
    for i in range(len(max_state)):
        if max_state[i] == 1:
            gumbel_perturb += perturbed_SAT.pos1_gumbel_perturbs[i]
        else:
            assert(max_state[i] == -1)
            gumbel_perturb += perturbed_SAT.neg1_gumbel_perturbs[i]
    return gumbel_perturb

def read_SAT_problem(problem_filename):
    '''
    Read in a SAT problem specified in WDIMACS format
    (http://www.maxhs.org/docs/wdimacs.html)

    Inputs:
    - problem_filename: string, the filename of the problem in WDIMACS format

    Outputs:
    - sat_problem: type SAT_problem, the SAT problem we read from problem_filename
    '''
    check_clause_count = 0
    clauses = []
    
    f = open(problem_filename, 'r')
    for line in f:
        if line[0] == 'p': #parameters line
            params = line.split()
            assert(params[1] == 'cnf') #we should be reading an unweighted SAT problem
            nbvar = int(params[2]) #number of variables
            nbclauses = int(params[3]) #number of clauses
        elif line[0] != 'c': #line isn't commented out, should be a clause
            clauses.append(line)
            check_clause_count += 1
        else:
            assert(line[0] == 'c')
    f.close()
    
    assert(check_clause_count == nbclauses), (check_clause_count, nbclauses) #did we read the correct number of clauses?
    sat_problem = SAT_problem(nbvar, nbclauses, clauses)
    return sat_problem

def write_SAT_problem(problem_filename, sat_problem, problem_type):
    '''
    Write a SAT problem to a file in WDIMACS format
    (http://www.maxhs.org/docs/wdimacs.html)

    Inputs:
        - problem_filename: string, the filename we will write the SAT problem to
        - sat_problem: type SAT_problem, the SAT problem we will write to problem_filename
        - problem_type: 'SAT' or 'weighted_MaxSat'

    Outputs:
        None, but we will write a file. Note that if a file with problem_filename exists,
        it will be erased and overwritten.
    '''
    assert(problem_type in ['SAT', 'weighted_MaxSat'])
    f = open(problem_filename, 'w')
    #write parameters line
    if problem_type == 'SAT':
        f.write("p cnf %d %d\n" % (sat_problem.nbvar, sat_problem.nbclauses))       
    else:
        f.write("p wcnf %d %d %d\n" % (sat_problem.nbvar, sat_problem.nbclauses, sat_problem.top))

    for clause in sat_problem.clauses:
        if problem_type == 'SAT':
            f.write(clause)
        else:
            f.write("%d %s" % (sat_problem.top, clause))

    if problem_type == 'weighted_MaxSat':
        if len(sat_problem.soft_clauses) > 0:
            f.write("\n")
        for soft_clause in sat_problem.soft_clauses:
            f.write(soft_clause)

    f.close()
    return None


def solve_SAT(sat_problem):
    '''
    Call a SAT solver to solve the specified SAT problem

    Inputs:
        - sat_problem: type SAT_problem, the SAT problem to solve
    Outputs:
        - satisying_solution: list of ints, each entry is either 1 or -1. 
        satisying_solution[i] is the value that variable i takes in the 
        satisfying solution to the SAT problem (1 indicates True, -1 False)
    '''
    satisying_solution = []

    write_SAT_problem('./temp_SAT_file.txt', sat_problem, problem_type='SAT')
    if SAT_SOLVER == "WBO":
        (status, output) = commands.getstatusoutput("%s/open-wbo_static ./temp_SAT_file.txt" % WBO_DIRECTORY)
    else:
        assert(SAT_SOLVER == "MAX_HS")
        (status, output) = commands.getstatusoutput("%s/maxhs ./temp_SAT_file.txt" % MAX_HS_DIRECTORY)

    for line in output.splitlines():
        if line[0] == 'v': #find the line in the output containing variable values in the solution
            params = line.split()
            assert(len(params) == sat_problem.nbvar + 1), (len(params), sat_problem.nbvar+1)
            for i in range(1, len(params)):
                if int(params[i]) > 0:
                    satisying_solution.append(1)
                else:
                    assert(int(params[i]) < 0)
                    satisying_solution.append(-1)

    return satisying_solution

def solve_weighted_partial_MaxSAT(sat_problem, time_limit, verbose=False):
    '''
    Call a SAT solver to solve the specified SAT problem

    Inputs:
    - sat_problem: type SAT_problem, the weighted partial MaxSAT
        problem to solve
    - time_limit: float, the number of seconds the SAT solver is given to solve the perturbed problem.
        If the problem is not solved in this time limit, we return None.
    - verbose: Bool, if true print the SAT solver's output

    Outputs:
    - max_solution: list of ints or None if the problem is not solved within the specified time limie.
        Each entry is either 1 or -1. max_solution[i] is the value that variable i takes in the 
        weighted partial MaxSAT solution
    '''
    max_solution = []

    write_SAT_problem('./temp_SAT_file.txt', sat_problem, problem_type='weighted_MaxSat')
    if SAT_SOLVER == "WBO":
#        (status, output) = commands.getstatusoutput("%s/open-wbo_static ./temp_SAT_file.txt" % WBO_DIRECTORY)        
        (status, output) = commands.getstatusoutput("perl -e 'alarm shift @ARGV; exec @ARGV' %f %s/open-wbo_static ./temp_SAT_file.txt" % (time_limit, WBO_DIRECTORY))
        if status != 0: #SAT solver failed to find a solution in the specifed time limit
            print 'SAT solver could not find a solution in %f seconds' % time_limit
            return None
    else:
        assert(SAT_SOLVER == "MAX_HS")
#        (status, output) = commands.getstatusoutput("%s/maxhs ./temp_SAT_file.txt" % MAX_HS_DIRECTORY)
        (status, output) = commands.getstatusoutput("perl -e 'alarm shift @ARGV; exec @ARGV' %f %s/maxhs ./temp_SAT_file.txt" % (time_limit, MAX_HS_DIRECTORY))
        if status != 0: #SAT solver failed to find a solution in the specifed time limit
            print 'SAT solver could not find a solution in %f seconds' % time_limit        
            return None
    
    if verbose:
        print output

    for line in output.splitlines():
        if len(line) > 0 and line[0] == 'v': #find the line in the output containing variable values in the solution
            var_solutions = line.split()
#           if len(var_solutions) != sat_problem.nbvar + 1:
#               print "something SKETCHY is happening with SAT solver, extra variable in solution"
#               var_solutions.pop() #discard last variable, again, SKETCHY
            assert(len(var_solutions) == sat_problem.nbvar + 1), (len(var_solutions), sat_problem.nbvar+1)

            for i in range(1, len(var_solutions)):
                if int(var_solutions[i]) > 0:
                    max_solution.append(1)
                else:
                    assert(int(var_solutions[i]) < 0)
                    max_solution.append(-1)

    print 'SAT solver found a solution within %f seconds' % time_limit
    return max_solution


def estimate_sharp_sat(sat_problem, gumbel_trials, k, chunk_size, \
    run_our_method=True, run_gumbel=True, run_new_gumbel_LB=True, 
    log_base=2.0, time_limit=99999999):
    '''
    Estimate and bound the number of satisfying assignments for the 
    specified SAT problem using gumbel and barvinok perturbations

    Inputs:
    - sat_problem: type SAT_problem, the weighted partial MaxSAT
        problem to solve
    - gumbel_trials: int, take the mean of this many max solutions to gumbel 
        perturbed problems for the gumbel upper bound
    - k: int, take the mean of k values of delta for our estimator
    - chunk_size: int, generate 2^chunk_size bits of randomness for every
        chunk_size block of the vector using our method
    - log_base: float, return all estimates, bounds, and log(Z) in this log base
    - time_limit: float, the number of seconds the SAT solver is given to solve the perturbed problems.  
        (for gumbel and our rademacher perturbations.)
        If a perturbed problem is not solved in this time limit, we sample new perturbations and try
        solving again.  This isn't ideal but some perturbations cause the SAT solver to get stuck 
        (quite sure this is true for Gumbel perturbations, not sure about Rademacher.  This might be
        because Gumbel perturbations are floats as opposed to integers, in which case another solution could
        be to round perturbations somehow but this would introduce additional approximation complexity.)

    Outputs:
    - estimators: dictionary with (key: values) of:
        'barv_estimator': list of 1 float, our estimator with barvinok perturbations
        'barv_upper': list of 1 float, our upper bound on #sat with barvinok perturbations
        'barv_lower': list of 1 float, our lower bound on #sat with barvinok perturbations
        'gumbel_upper': list of 1 float, upper bound on #sat with gumberl perturbations
    '''
    print 'a'
    t0 = time.time()
    list_of_deltas = []
    if run_our_method:
        for i in range(k):
            barv_max_solution = None
            #generate perturbed SAT problems until we find a solution within the specified time limit
            while(barv_max_solution==None):
                barv_pert_sat_problem = copy.deepcopy(sat_problem)
                if chunk_size == None:
                    barv_pert_sat_problem.barvinok_perturb()
                else:
                    barv_pert_sat_problem.barvinok_perturb_chunked(chunk_size=chunk_size)
                barv_max_solution = solve_weighted_partial_MaxSAT(barv_pert_sat_problem, time_limit)
                if barv_max_solution != None: #we found a solution to the perturbed SAT problem within the specified time limit
                    delta = get_delta(barv_pert_sat_problem, barv_max_solution)
                    list_of_deltas.append(delta)
                    #assert(delta > 0), ("delta negative!", delta)
        delta_bar_median = np.median(list_of_deltas)
        delta_bar_mean = np.mean(list_of_deltas)
#        #delta estimates log_2(Z), rescale to estimate ln(Z)
#        delta_bar *= np.log(2)

#    (barv_high_prob_UB, barv_expectation_UB) = gen_upper_bound_Z(delta_bar=delta_bar, n=sat_problem.nbvar, k=k, log_base=np.e, w_max=1, w_min=1, verbose=True, chunk_size=chunk_size)

    print "1111debugingdebugingdebuging, delta_bar_mean =", delta_bar_mean
    (barv_high_prob_UB, barv_expectation_UB) = gen_upper_bound_Z(delta_bar=delta_bar_mean, n=sat_problem.nbvar, k=k, log_base=log_base, w_max=1, w_min=1, verbose=True)
    (barv_high_prob_LB, barv_expectation_LB) = gen_lower_bound_Z(delta_bar=delta_bar_mean, n=sat_problem.nbvar, k=k, log_base=log_base, w_max=1, w_min=1, verbose=True)
    print "2222debugingdebugingdebuging, delta_bar_mean =", delta_bar_mean
 
    t1 = time.time()

######    #delta estimates log_2(Z), rescale to estimate ln(Z)
######    delta_bar *= np.log(2)
    #delta estimates log_2(Z), rescale to estimate log(Z) in base log_base
    delta_bar_mean = delta_bar_mean/(np.log(log_base)/np.log(2.0))


    print 'b'
    t2 = time.time()    
    if run_gumbel:
        gumbel_expectation_UB = 0.0
        for i in range(gumbel_trials):
            gumbel_max_solution = None
            #generate perturbed SAT problems until we find a solution within the specified time limit
            while(gumbel_max_solution==None):

                gumbel_pert_sat_problem = copy.deepcopy(sat_problem)
                gumbel_pert_sat_problem.gumbel_perturb()
                gumbel_max_solution = solve_weighted_partial_MaxSAT(gumbel_pert_sat_problem, time_limit)
                if gumbel_max_solution != None: #we found a solution to the perturbed SAT problem within the specified time limit
                    max_gumbel_perturbation = get_gumbel_perturbed_max(gumbel_pert_sat_problem, gumbel_max_solution)
                    gumbel_expectation_UB += max_gumbel_perturbation

        gumbel_expectation_UB /= gumbel_trials
        gumbel_slack = calculate_gumbel_slack(A=sat_problem.nbvar, M=gumbel_trials, delta=.05)
        gumbel_high_prob_UB = gumbel_expectation_UB + gumbel_slack

        gumbel_expectation_LB = gumbel_expectation_UB/sat_problem.nbvar
        gumbel_high_prob_LB = (gumbel_expectation_UB - gumbel_slack)/sat_problem.nbvar

        #gumbel bounds are in log base e, rescale to base log_base
        gumbel_expectation_UB = gumbel_expectation_UB/np.log(log_base)
        gumbel_high_prob_UB = gumbel_high_prob_UB/np.log(log_base)
        gumbel_expectation_LB = gumbel_expectation_LB/np.log(log_base)
        gumbel_high_prob_LB = gumbel_high_prob_LB/np.log(log_base)

    else:
        gumbel_expectation_UB = 0.0
        gumbel_high_prob_UB = 0.0
        gumbel_expectation_LB = 0.0
        gumbel_high_prob_LB = 0.0
    t3 = time.time()        


    massart_gumbelPerturb_LB = None
    massart_gumbelPerturb_LB_lambda = None
    massart_gumbelPerturb_LB_expectation_est = None
    massart_gumbelPerturb_LB_lambda_expectation_est = None    
    delta_bar_g_mean = None
    delta_bar_g_median = None
    slack_CHECK_ME = None
    gumbel_location=-np.euler_gamma
    if run_new_gumbel_LB: #described here: https://www.sharelatex.com/read/kvkmnmxmpzct
        list_of_deltas_g = []
        for i in range(k):
            barv_max_solution_g = None
            #generate perturbed SAT problems until we find a solution within the specified time limit
            while(barv_max_solution_g==None):
                barv_pert_sat_problem_g = copy.deepcopy(sat_problem)
                barv_pert_sat_problem_g.dot_product_perturb_gumbel(gumbel_location=gumbel_location)
                barv_max_solution_g = solve_weighted_partial_MaxSAT(barv_pert_sat_problem_g, time_limit)
                if barv_max_solution_g != None: #we found a solution to the perturbed SAT problem within the specified time limit
                    delta_g = get_delta(barv_pert_sat_problem_g, barv_max_solution_g)
                    list_of_deltas_g.append(delta_g)
                    #assert(delta_g > 0), ("delta_g negative!", delta_g)
        delta_bar_g_median = np.median(list_of_deltas_g)
        delta_bar_g_mean = np.mean(list_of_deltas_g)


#        slack_CHECK_ME = calculate_gumbel_slack(A=sat_problem.nbvar, M=k, delta=.05)
        #this slack bound is incorrect, but using to check giving this method an advantage
        slack_CHECK_ME = np.sqrt(6*sat_problem.nbvar/k)

        if gumbel_location == 0:
            lambda_vals = [1/1000, 1/100, 1/10, 1/4, 1/2, 1]
            log2_E_lambda_vals = [0.000578, 0.005832, 0.063628, 0.183357, 0.462922, 1.546685]

        else:
            assert(gumbel_location == -np.euler_gamma) #gumbel ditribution shifted to have mean 0
            lambda_vals = [1, .7, 1/2, 1/4, 1/10, 1/100]
            log2_E_lambda_vals = [0.9694691807230248, 0.3805521478961128, 0.17431692647381947, 0.03904010476533171, 0.005902956858236386, 0.00005770664751036936]
        assert(len(lambda_vals) == len(log2_E_lambda_vals))
        massart_gumbelPerturb_LB = None
        for lambda_val, log2_E_lambda_val in zip(lambda_vals, log2_E_lambda_vals):
            cur_lower_bound = lambda_val * delta_bar_g_mean - lambda_val * slack_CHECK_ME - sat_problem.nbvar * log2_E_lambda_val
            if massart_gumbelPerturb_LB == None or cur_lower_bound > massart_gumbelPerturb_LB:
                massart_gumbelPerturb_LB = cur_lower_bound
                massart_gumbelPerturb_LB_lambda = lambda_val

            cur_lower_bound_expectation_estimate = lambda_val * delta_bar_g_mean - sat_problem.nbvar * log2_E_lambda_val
            if massart_gumbelPerturb_LB_expectation_est == None or cur_lower_bound_expectation_estimate > massart_gumbelPerturb_LB_expectation_est:
                massart_gumbelPerturb_LB_expectation_est = cur_lower_bound_expectation_estimate
                massart_gumbelPerturb_LB_lambda_expectation_est = lambda_val

        #estimate and bound are in log base 2, rescale to base log_base
        delta_bar_g_mean = delta_bar_g_mean/(np.log(log_base)/np.log(2.0))
        massart_gumbelPerturb_LB = massart_gumbelPerturb_LB/(np.log(log_base)/np.log(2.0))


    print 'c'
    print 'massart_gumbelPerturb_LB =', massart_gumbelPerturb_LB
    print 'massart_gumbelPerturb_LB_lambda =', massart_gumbelPerturb_LB_lambda
    print '-'*80
    estimators = {'barv_estimator': [delta_bar_mean],
                  'barv_estimator_median': [delta_bar_median],
                  'barv_high_prob_UB': [barv_high_prob_UB],
                  'barv_expectation_UB': [barv_expectation_UB],
                  'barv_high_prob_LB': [barv_high_prob_LB],
                  'barv_expectation_LB': [barv_expectation_LB],
                  'barv_compute_time': [t1 - t0], #time used to compute barvinok upper and lower bounds
                  'gumb_compute_time': [t3 - t2], #time used to compute gumbel upper and lower bounds
                  'gumbel_expectation_UB': [gumbel_expectation_UB],
                  'gumbel_high_prob_UB': [gumbel_high_prob_UB],
                  'gumbel_expectation_LB': [gumbel_expectation_LB],
                  'gumbel_high_prob_LB': [gumbel_high_prob_LB],
                  'massart_gumbelPerturb_LB': [massart_gumbelPerturb_LB],
                  'massart_gumbelPerturb_LB_lambda': [massart_gumbelPerturb_LB_lambda],
                  'massart_gumbelPerturb_LB_expectation_est': [massart_gumbelPerturb_LB_expectation_est],
                  'massart_gumbelPerturb_LB_lambda_expectation_est': [massart_gumbelPerturb_LB_lambda_expectation_est],
                  'delta_k_gumbel_perturb': [delta_bar_g_mean],
                  'delta_k_gumbel_perturb_median': [delta_bar_g_median],
                  'epsilon_g': [slack_CHECK_ME],}



    return estimators


def save_results_data(data_filename, results, num_gumbel_pert, our_k, log_base):
    f = open(data_filename, 'w')
    pickle.dump((results, num_gumbel_pert, our_k, log_base), f)
    f.close() 

def load_results_data(data_filename):
    f = open(data_filename, 'r')
    (results, num_gumbel_pert, our_k, log_base) = pickle.load(f)
    f.close()
    return (results, num_gumbel_pert, our_k, log_base)

def print_results_latex_table(results, num_gumbel_pert, our_k, log_base):
    '''
    print latex table of our results

    Inputs:
    - results: dictionary, with-
        key: string, model_txt_file
        value: dictionary with-
           key: string, the estimator 'barv_estimator', 'barv_high_prob_UB', 'barv_expectation_UB', 'barv_high_prob_LB', 'barv_expectation_LB', 'gumbel_expectation_UB', 'gumbel_high_prob_UB', 'gumbel_expectation_LB', 'gumbel_high_prob_LB'
           value: float, the value of the estimator
    '''
    var_counts = {
                    "log-1.cnf" : 939,
                    "log-2.cnf" : 1337,
                    "log-3.cnf" : 1413,
                    "log-4.cnf" : 2303,
                    "log-5.cnf" : 2701,
                    "tire-1.cnf" : 352,
                    "tire-2.cnf" : 550,
                    "tire-3.cnf" : 577,
                    "tire-4.cnf" : 812,
                    "ra.cnf" : 1236,
                    "rb.cnf" : 1854,
                    "rc.cnf" : 2472,
                    "sat-grid-pbl-0010.cnf" : 110,
                    "sat-grid-pbl-0015.cnf" : 240,
                    "sat-grid-pbl-0020.cnf" : 420,
                    "sat-grid-pbl-0025.cnf" : 650,
                    "sat-grid-pbl-0030.cnf" : 930,
                    "c432.isc" : 196,
                    "c499.isc" : 243,
                    "c880.isc" : 417,
                    "c1355.isc" : 555,
                    "c1908.isc" : 751,
                    "c2670.isc" : 1230,
                    "c7552.isc" : 3185,
                    "lang12.cnf" : 576,
                    "wff.3.150.525.cnf" : 150,
                }

    clause_counts = {
                    "log-1.cnf" : 3785,
                    "log-2.cnf" : 24777,
                    "log-3.cnf" : 29487,
                    "log-4.cnf" : 20963,
                    "log-5.cnf" : 29534,
                    "tire-1.cnf" : 1038,
                    "tire-2.cnf" : 2001,
                    "tire-3.cnf" : 2004,
                    "tire-4.cnf" : 3222,
                    "ra.cnf" : 11416,
                    "rb.cnf" : 11324,
                    "rc.cnf" : 17942,
                    "sat-grid-pbl-0010.cnf" : 191,
                    "sat-grid-pbl-0015.cnf" : 436,
                    "sat-grid-pbl-0020.cnf" : 781,
                    "sat-grid-pbl-0025.cnf" : 1226,
                    "sat-grid-pbl-0030.cnf" : 1771,
                    "c432.isc" : 514,
                    "c499.isc" : 714,
                    "c880.isc" : 1060,
                    "c1355.isc" : 1546,
                    "c1908.isc" : 2053,
                    "c2670.isc" : 2876,
                    "c7552.isc" : 8588,
                    "lang12.cnf" : 576,      
                    "wff.3.150.525.cnf" : 150,
                }
    #all model names
#    model_names = ["log-1.cnf", "log-2.cnf", "log-3.cnf", "log-4.cnf", "log-5.cnf", "tire-1.cnf", "tire-2.cnf", "tire-3.cnf", "tire-4.cnf", "ra.cnf", "rb.cnf", "rc.cnf", "sat-grid-pbl-0010.cnf", "sat-grid-pbl-0015.cnf", "sat-grid-pbl-0020.cnf", "sat-grid-pbl-0025.cnf", "sat-grid-pbl-0030.cnf", "c432.isc", "c499.isc", "c880.isc", "c1355.isc", "c1908.isc", "c2670.isc", "c7552.isc"]
    
    #exclude some models
    model_names = ["log-1.cnf", "log-2.cnf", "log-3.cnf", "log-4.cnf", "tire-1.cnf", "tire-2.cnf", "tire-3.cnf", "tire-4.cnf", "ra.cnf", "rb.cnf", "sat-grid-pbl-0010.cnf", "sat-grid-pbl-0015.cnf", "sat-grid-pbl-0020.cnf", "sat-grid-pbl-0025.cnf", "sat-grid-pbl-0030.cnf", "c432.isc", "c499.isc", "c880.isc", "c1355.isc", "c1908.isc", "c2670.isc"]
#    model_names = ["lang12.cnf", "wff.3.150.525.cnf"]
#    model_names = ["wff.3.150.525.cnf"]
#    model_names = ["lang12.cnf"]

#    model_names = ["sat-grid-pbl-0010.cnf", "sat-grid-pbl-0015.cnf", "sat-grid-pbl-0020.cnf", "sat-grid-pbl-0025.cnf", "sat-grid-pbl-0030.cnf"]

    print '-'*30, "High probability bounds:", '-'*30
    print "Model Name & \#Var & \#Claus & log(Z) & Our Est & Our UB & Gumbel UB & Our LB & Gumbel LB & Our LB gumbel perturb & lambda & Our Compute Time & Gumbel Compute Time\\\\ \hline"
    for name in model_names:
        cur_est = results[name]

####        ## DEBUG ##
####        print  np.mean(var_counts[name])
####        print  np.mean(clause_counts[name])
####        print cur_est['exact_log_Z']
####        print '!!!!!!!!'
####        print  np.mean(cur_est['exact_log_Z'])
####        print  np.std(cur_est['exact_log_Z'])
####        print  np.mean(cur_est['barv_estimator'])
####        print  np.std(cur_est['barv_estimator'])
####        ## END DEBUG ##

        print "     %s             & %d                  & %d                & %.1f                   & %.1f        (%.1f)         " \
            % (name, np.mean(var_counts[name]), np.mean(clause_counts[name]), np.mean(cur_est['exact_log_Z']), np.mean(cur_est['barv_estimator']), np.std(cur_est['barv_estimator'])),

        if np.mean(cur_est['barv_high_prob_UB']) < np.mean(cur_est['gumbel_high_prob_UB']):
            print "& \\textbf{%.1f}      (%.1f)      & %.1f        (%.1f)      " % (np.mean(cur_est['barv_high_prob_UB']),np.std(cur_est['barv_high_prob_UB']), np.mean(cur_est['gumbel_high_prob_UB']),np.std(cur_est['gumbel_high_prob_UB'])),
        else:
            print "& %.1f       (%.1f)     & \\textbf{%.1f}        (%.1f)      " % (np.mean(cur_est['barv_high_prob_UB']), np.std(cur_est['barv_high_prob_UB']), np.mean(cur_est['gumbel_high_prob_UB']), np.std(cur_est['gumbel_high_prob_UB'])),


        if np.mean(cur_est['barv_high_prob_LB']) > np.mean(cur_est['gumbel_high_prob_LB']):
            print "& \\textbf{%.1f}     (%.1f)       & %.1f      (%.1f)     & %.1f      (%.1f)     & %.1f       & %.3f  (%.3f)& %.3f  (%.3f)\\\\ \\hline" % (np.mean(cur_est['barv_high_prob_LB']), np.std(cur_est['barv_high_prob_LB']), np.mean(cur_est['gumbel_high_prob_LB']), np.std(cur_est['gumbel_high_prob_LB']), np.mean(cur_est['massart_gumbelPerturb_LB']), np.std(cur_est['massart_gumbelPerturb_LB']), np.mean(cur_est['massart_gumbelPerturb_LB_lambda']), np.mean(cur_est['barv_compute_time']), np.std(cur_est['barv_compute_time']), np.mean(cur_est['gumb_compute_time']), np.std(cur_est['gumb_compute_time'])),
        else:
            print "& %.1f       (%.1f)     & \\textbf{%.1f}       (%.1f)    & %.1f      (%.1f)     & %.1f       & %.3f  (%.3f)& %.3f  (%.3f)\\\\ \\hline" % (np.mean(cur_est['barv_high_prob_LB']), np.std(cur_est['barv_high_prob_LB']), np.mean(cur_est['gumbel_high_prob_LB']), np.std(cur_est['gumbel_high_prob_LB']), np.mean(cur_est['massart_gumbelPerturb_LB']), np.std(cur_est['massart_gumbelPerturb_LB']), np.mean(cur_est['massart_gumbelPerturb_LB_lambda']), np.mean(cur_est['barv_compute_time']), np.std(cur_est['barv_compute_time']), np.mean(cur_est['gumb_compute_time']), np.std(cur_est['gumb_compute_time'])),
    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{LOG BASE = %f, gumbel k = %d, our k = %d}" % (log_base, num_gumbel_pert, our_k)
    
    print
 
    print '-'*30, "Expectation bounds:", '-'*30
    print "Model Name & \#Var & \#Claus & log(Z) & Our Est & Our UB & Gumbel UB & Our LB & Gumbel LB \\\\ \hline"
    for name in model_names:
        cur_est = results[name]
        print "     %s             & %d                  & %d                & %.1f                    & %.1f                 " \
            % (name, np.mean(var_counts[name]), np.mean(clause_counts[name]), np.mean(cur_est['exact_log_Z']), np.mean(cur_est['barv_estimator'])),

        if np.mean(cur_est['barv_expectation_UB']) < np.mean(cur_est['gumbel_expectation_UB']):
            print "& \\textbf{%.1f}            & %.1f              " % (np.mean(cur_est['barv_expectation_UB']), np.mean(cur_est['gumbel_expectation_UB'])),
        else:
            print "& %.1f            & \\textbf{%.1f}              " % (np.mean(cur_est['barv_expectation_UB']), np.mean(cur_est['gumbel_expectation_UB'])),


        if np.mean(cur_est['barv_expectation_LB']) > np.mean(cur_est['gumbel_expectation_LB']):
            print "& \\textbf{%.1f}            & %.1f              \\\\ \\hline" % (np.mean(cur_est['barv_expectation_LB']), np.mean(cur_est['gumbel_expectation_LB'])),
        else:
            print "& %.1f            & \\textbf{%.1f}              \\\\ \\hline" % (np.mean(cur_est['barv_expectation_LB']), np.mean(cur_est['gumbel_expectation_LB'])),

    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{LOG BASE = %f, gumbel k = %d, our k = %d}" % (log_base, num_gumbel_pert, our_k)


    print '-'*30, "Bound width:", '-'*30
    print "Model Name & \#Var & \#Claus & log(Z) & Our HP width & Gumbel HP width & Our Exp width & Gumbel Exp width  \\\\ \hline"
    for name in model_names:
        cur_est = results[name]

        print "     %s             & %d                  & %d                & %.1f                    " \
            % (name, var_counts[name], clause_counts[name], cur_est['exact_log_Z']),

        if cur_est['barv_high_prob_UB'] - cur_est['barv_high_prob_LB'] < \
           cur_est['gumbel_high_prob_UB'] - cur_est['gumbel_high_prob_LB']:
            print "& \\textbf{%.1f}                 & %.1f            " % \
                (cur_est['barv_high_prob_UB'] - cur_est['barv_high_prob_LB'],
                cur_est['gumbel_high_prob_UB'] - cur_est['gumbel_high_prob_LB']),
        else:          
            print "& %.1f                 & \\textbf{%.1f}            " % \
                (cur_est['barv_high_prob_UB'] - cur_est['barv_high_prob_LB'],
                cur_est['gumbel_high_prob_UB'] - cur_est['gumbel_high_prob_LB']),

        if cur_est['barv_expectation_UB'] - cur_est['barv_expectation_LB'] < \
           cur_est['gumbel_expectation_UB'] - cur_est['gumbel_expectation_LB']:
            print "& \\textbf{%.1f}              & %.1f            \\\\ \\hline" % \
                (cur_est['barv_expectation_UB'] - cur_est['barv_expectation_LB'],
                 cur_est['gumbel_expectation_UB'] - cur_est['gumbel_expectation_LB'])
        else:
            print "& %.1f              & \\textbf{%.1f}            \\\\ \\hline" % \
                (cur_est['barv_expectation_UB'] - cur_est['barv_expectation_LB'],
                 cur_est['gumbel_expectation_UB'] - cur_est['gumbel_expectation_LB'])

    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{LOG BASE = %f, gumbel k = %d, our k = %d}" % (log_base, num_gumbel_pert, our_k)

def print_results_latex_table_ourGumbelPerturbLB(results, num_gumbel_pert, our_k, log_base):
    '''
    print latex table of our results

    Inputs:
    - results: dictionary, with-
        key: string, model_txt_file
        value: dictionary with-
           key: string, the estimator 'barv_estimator', 'barv_high_prob_UB', 'barv_expectation_UB', 'barv_high_prob_LB', 'barv_expectation_LB', 'gumbel_expectation_UB', 'gumbel_high_prob_UB', 'gumbel_expectation_LB', 'gumbel_high_prob_LB'
           value: float, the value of the estimator
    '''
    var_counts = {
                    "log-1.cnf" : 939,
                    "log-2.cnf" : 1337,
                    "log-3.cnf" : 1413,
                    "log-4.cnf" : 2303,
                    "log-5.cnf" : 2701,
                    "tire-1.cnf" : 352,
                    "tire-2.cnf" : 550,
                    "tire-3.cnf" : 577,
                    "tire-4.cnf" : 812,
                    "ra.cnf" : 1236,
                    "rb.cnf" : 1854,
                    "rc.cnf" : 2472,
                    "sat-grid-pbl-0010.cnf" : 110,
                    "sat-grid-pbl-0015.cnf" : 240,
                    "sat-grid-pbl-0020.cnf" : 420,
                    "sat-grid-pbl-0025.cnf" : 650,
                    "sat-grid-pbl-0030.cnf" : 930,
                    "c432.isc" : 196,
                    "c499.isc" : 243,
                    "c880.isc" : 417,
                    "c1355.isc" : 555,
                    "c1908.isc" : 751,
                    "c2670.isc" : 1230,
                    "c7552.isc" : 3185,
                    "lang12.cnf" : 576,
                    "wff.3.150.525.cnf" : 150,
                }

    clause_counts = {
                    "log-1.cnf" : 3785,
                    "log-2.cnf" : 24777,
                    "log-3.cnf" : 29487,
                    "log-4.cnf" : 20963,
                    "log-5.cnf" : 29534,
                    "tire-1.cnf" : 1038,
                    "tire-2.cnf" : 2001,
                    "tire-3.cnf" : 2004,
                    "tire-4.cnf" : 3222,
                    "ra.cnf" : 11416,
                    "rb.cnf" : 11324,
                    "rc.cnf" : 17942,
                    "sat-grid-pbl-0010.cnf" : 191,
                    "sat-grid-pbl-0015.cnf" : 436,
                    "sat-grid-pbl-0020.cnf" : 781,
                    "sat-grid-pbl-0025.cnf" : 1226,
                    "sat-grid-pbl-0030.cnf" : 1771,
                    "c432.isc" : 514,
                    "c499.isc" : 714,
                    "c880.isc" : 1060,
                    "c1355.isc" : 1546,
                    "c1908.isc" : 2053,
                    "c2670.isc" : 2876,
                    "c7552.isc" : 8588,
                    "lang12.cnf" : 576,      
                    "wff.3.150.525.cnf" : 150,
                }
    #all model names
#    model_names = ["log-1.cnf", "log-2.cnf", "log-3.cnf", "log-4.cnf", "log-5.cnf", "tire-1.cnf", "tire-2.cnf", "tire-3.cnf", "tire-4.cnf", "ra.cnf", "rb.cnf", "rc.cnf", "sat-grid-pbl-0010.cnf", "sat-grid-pbl-0015.cnf", "sat-grid-pbl-0020.cnf", "sat-grid-pbl-0025.cnf", "sat-grid-pbl-0030.cnf", "c432.isc", "c499.isc", "c880.isc", "c1355.isc", "c1908.isc", "c2670.isc", "c7552.isc"]
    
    #exclude some models
#   probably want this list of model_names:
    model_names = ["log-1.cnf", "log-2.cnf", "log-3.cnf", "log-4.cnf", "tire-1.cnf", "tire-2.cnf", "tire-3.cnf", "tire-4.cnf", "ra.cnf", "rb.cnf", "sat-grid-pbl-0010.cnf", "sat-grid-pbl-0015.cnf", "sat-grid-pbl-0020.cnf", "sat-grid-pbl-0025.cnf", "sat-grid-pbl-0030.cnf", "c432.isc", "c499.isc", "c880.isc", "c1355.isc", "c1908.isc", "c2670.isc"]

#    model_names = ["rb.cnf"]

#    model_names = ["lang12.cnf", "wff.3.150.525.cnf"]
#    model_names = ["wff.3.150.525.cnf"]
#    model_names = ["lang12.cnf"]

#    model_names = ["sat-grid-pbl-0010.cnf", "sat-grid-pbl-0015.cnf", "sat-grid-pbl-0020.cnf", "sat-grid-pbl-0025.cnf", "sat-grid-pbl-0030.cnf"]

    print '-'*30, "High probability bounds:", '-'*30
    print "Model Name & \#Var & \#Claus & log(Z) & Our Est & Our UB &  & Our LB &  & Our LB G & $\lambda$ & $\delta_g$ & $\epsilon_g$ & LB exp est & $\lambda$\\\\ \hline"
    for name in model_names:
        cur_est = results[name]

        print "     %s             & %d                  & %d                & %.1f                   & %.1f                 " \
            % (name, np.mean(var_counts[name]), np.mean(clause_counts[name]), np.mean(cur_est['exact_log_Z']), np.mean(cur_est['barv_estimator'])),

        if np.mean(cur_est['barv_high_prob_UB']) < np.mean(cur_est['gumbel_high_prob_UB']):
            print "& \\textbf{%.1f}            & %.1f        " % (np.mean(cur_est['barv_high_prob_UB']), np.mean(cur_est['gumbel_high_prob_UB'])),
        else:
            print "& %.1f      & \\textbf{%.1f}        " % (np.mean(cur_est['barv_high_prob_UB']),  np.mean(cur_est['gumbel_high_prob_UB'])),


        if np.mean(cur_est['barv_high_prob_LB']) > np.mean(cur_est['gumbel_high_prob_LB']):
            print "& \\textbf{%.1f}       & %.1f     & %.1f     & %.3f       & %.3f  & %.3f & %.1f     & %.3f \\\\ \\hline" % (np.mean(cur_est['barv_high_prob_LB']),  np.mean(cur_est['gumbel_high_prob_LB']),  np.mean(cur_est['massart_gumbelPerturb_LB']),  np.mean(cur_est['massart_gumbelPerturb_LB_lambda']), np.mean(cur_est['delta_k_gumbel_perturb']),  np.mean(cur_est['epsilon_g']), np.mean(cur_est['massart_gumbelPerturb_LB_expectation_est']),  np.mean(cur_est['massart_gumbelPerturb_LB_lambda_expectation_est'])),
        else:
            print "& %.1f      & \\textbf{%.1f}     & %.1f     & %.3f       & %.3f  & %.3f & %.1f     & %.3f \\\\ \\hline" % (np.mean(cur_est['barv_high_prob_LB']),  np.mean(cur_est['gumbel_high_prob_LB']),  np.mean(cur_est['massart_gumbelPerturb_LB']),  np.mean(cur_est['massart_gumbelPerturb_LB_lambda']), np.mean(cur_est['delta_k_gumbel_perturb']),  np.mean(cur_est['epsilon_g']), np.mean(cur_est['massart_gumbelPerturb_LB_expectation_est']),  np.mean(cur_est['massart_gumbelPerturb_LB_lambda_expectation_est'])),


    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{LOG BASE = %f, gumbel k = %d, our k = %d}" % (log_base, num_gumbel_pert, our_k)


    print
    print '-'*30, "High probability bounds:", '-'*30
    print "Model Name & \#Var & \#Claus & log(Z) & Our Est (mean)& Our Est (median)& Our UB & Gumbel UB & Our LB & Gumbel LB & Our LB G & $\lambda$ & $\delta_g$ (mean) & $\delta_g$ (median) & $\epsilon_g$\\\\ \hline"
    for name in model_names:
        cur_est = results[name]

        print "     %s             & %d                  & %d                & %.1f                   & %.1f                 & %.1f                 " \
            % (name, np.mean(var_counts[name]), np.mean(clause_counts[name]), np.mean(cur_est['exact_log_Z']), np.mean(cur_est['barv_estimator']), np.mean(cur_est['barv_estimator_median'])),

        if np.mean(cur_est['barv_high_prob_UB']) < np.mean(cur_est['gumbel_high_prob_UB']):
            print "& \\textbf{%.1f}   (%.1f)         & %.1f    (%.1f)    " % (np.mean(cur_est['barv_high_prob_UB']), np.std(cur_est['barv_high_prob_UB']), np.mean(cur_est['gumbel_high_prob_UB']), np.std(cur_est['gumbel_high_prob_UB'])),
        else:
            print "& %.1f   (%.1f)   & \\textbf{%.1f}    (%.1f)    " % (np.mean(cur_est['barv_high_prob_UB']), np.std(cur_est['barv_high_prob_UB']),  np.mean(cur_est['gumbel_high_prob_UB']), np.std(cur_est['gumbel_high_prob_UB'])),


        if np.mean(cur_est['barv_high_prob_LB']) > np.mean(cur_est['gumbel_high_prob_LB']):
            print "& \\textbf{%.1f}    (%.1f)   & %.1f   (%.1f)  & %.1f  (%.1f)   & %.4f       & %.3f  & %.3f & %.3f  \\\\ \\hline" % (np.mean(cur_est['barv_high_prob_LB']), np.std(cur_est['barv_high_prob_LB']),  np.mean(cur_est['gumbel_high_prob_LB']), np.std(cur_est['gumbel_high_prob_LB']),  np.mean(cur_est['massart_gumbelPerturb_LB']), np.std(cur_est['massart_gumbelPerturb_LB']),  np.mean(cur_est['massart_gumbelPerturb_LB_lambda']), np.mean(cur_est['delta_k_gumbel_perturb']), np.mean(cur_est['delta_k_gumbel_perturb_median']), np.mean(cur_est['epsilon_g'])),
        else:
            print "& %.1f   (%.1f)   & \\textbf{%.1f}  (%.1f)   & %.1f   (%.1f)  & %.4f       & %.3f  & %.3f & %.3f  \\\\ \\hline" % (np.mean(cur_est['barv_high_prob_LB']), np.std(cur_est['barv_high_prob_LB']),  np.mean(cur_est['gumbel_high_prob_LB']), np.std(cur_est['gumbel_high_prob_LB']),  np.mean(cur_est['massart_gumbelPerturb_LB']), np.std(cur_est['massart_gumbelPerturb_LB']),  np.mean(cur_est['massart_gumbelPerturb_LB_lambda']), np.mean(cur_est['delta_k_gumbel_perturb']), np.mean(cur_est['delta_k_gumbel_perturb_median']), np.mean(cur_est['epsilon_g'])),

    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{LOG BASE = %f, gumbel k = %d, our k = %d}" % (log_base, num_gumbel_pert, our_k)

    print

if __name__=="__main__":
    LOG_BASE = 2.0 #use this log base for estimates and bounds
    NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS = 10
    OUR_K = 10
    CHUNK_SIZE = None

    RUN_SPECIAL = False

    #perform this many experiments so we can compare bound statistics
    NUM_EXPERIMENTS = 1

    #the number of seconds the SAT solver is given to solve the perturbed problems.  
    TIME_LIMIT = 10.0
############################ PRINT LATEX TABLE OF STORED RESULTS ############################
    ONLY_PRINT_LATEX_TABLE = False #print latex table from stored results without running new experiment
    if ONLY_PRINT_LATEX_TABLE:
        if RUN_SPECIAL:
            if CHUNK_SIZE:
                data_filename = "/atlas/u/jkuck/AAAI_2017/refactored_multi_model/results_data/lang12_sat_results_k=%d_gumbelPerturbCount=%d_chunk_size=%dnumExp=%d" % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, CHUNK_SIZE, NUM_EXPERIMENTS)
            else:
                data_filename = "/atlas/u/jkuck/AAAI_2017/refactored_multi_model/results_data/lang12_sat_results_k=%d_gumbelPerturbCount=%d_chunk_size=NonenumExp=%d" % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, NUM_EXPERIMENTS)


        else:        
            if CHUNK_SIZE:
                data_filename = "/atlas/u/jkuck/AAAI_2017/refactored_multi_model/results_data/sat_results_k=%d_gumbelPerturbCount=%d_chunk_size=%dnumExp=%d" % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, CHUNK_SIZE, NUM_EXPERIMENTS)
            else:
                data_filename = "/atlas/u/jkuck/AAAI_2017/refactored_multi_model/results_data/sat_results_k=%d_gumbelPerturbCount=%d_chunk_size=NonenumExp=%d" % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, NUM_EXPERIMENTS)

        (loaded_results, num_gumbel_pert, our_k, log_base) = load_results_data(data_filename)
#        print_results_latex_table(loaded_results, num_gumbel_pert, our_k, log_base)
        print_results_latex_table_ourGumbelPerturbLB(loaded_results, num_gumbel_pert, our_k, log_base)
        sleep(43)

    #sat_problem = read_SAT_problem("../SAT_problems/SAT_problems_cnf/%s" % "c499.isc")
    #Z = 593962746002226256572855
    #write_SAT_problem('temp_SAT_file.txt', sat_problem)
    #solve_SAT(sat_problem, problem_type='SAT')


    if not RUN_SPECIAL:
        RUN_MODEL_SUBSET = False
        if RUN_MODEL_SUBSET:
            model_counts = {
                "sat-grid-pbl-0010.cnf" : 593962746002226256572855,
                "sat-grid-pbl-0015.cnf" : 3012964503482414330783936367006634039953704207876657607,
                "sat-grid-pbl-0020.cnf" : 505529009203800755681036389586231130590495376930061598744188591803177007627164988944437560726719,
                "sat-grid-pbl-0025.cnf" : 18051755963842913831469740173876709710679368179331841451265906666609267268260641283484985491991777996404498069889539875259775880457350865091313349862141,
                "sat-grid-pbl-0030.cnf" : 154089430409193243326541334620745040441316292191902104056995281410078040886572580007404908279631836580121336881940079952338956656599622521005239507469477568002534440349139077306892061020210022834318422387583588123648727
            }
        else:
            #dictionary with key: values of model_name: number of satisfying solutions
            model_counts = {
                "c880.isc" : 1152921504606846976,

                "log-1.cnf" : 564153552511417968750,
                "log-2.cnf" : 32334741710,
                "log-3.cnf" : 279857462060,
                "log-4.cnf" : 23421510324076617565622131248,
        #        "log-5.cnf" : 724152621485436659540387630662916505600,
                "tire-1.cnf": 726440820,
                "tire-2.cnf": 738969640920,
                "tire-3.cnf": 222560409176,
                "tire-4.cnf": 103191650628000,
                "ra.cnf" : 18739277038847939886754019920358123424308469030992781557966909983211910963157763678726120154469030856807730587971859910379069462105489708001873004723798633342340521799560185957916958401869207109443355859123561156747098129524433371596461424856004227854241384374972430825095073282950873641,
                "rb.cnf" : 538812462750928282721716308734898413194103864553832956073815148020987917365241105816807625188941823391012398976196112157887068449390989186368113511090801920577156367304804512491926215360520651047719401241944845883406098779319363314639309655779343140788696509960447536163784266937815828202118895534508004061478961506257883130142920912100543747226035966976598909666696626176,
        #        "rc.cnf" : 7711354164502494341321469992586296215389019368675210425477306574721979716330369834365339212526497517240811559116147742518536494403648202692367307372347375124735195570200184068550084511307308544710567552927267754358098889434805018720727997741618028500536976980307056282336258719799038253686515232663203157545908220322265455973957544442057437359833452997837270984970131866759705201073382727090176,
                "sat-grid-pbl-0010.cnf" : 593962746002226256572855,
                "sat-grid-pbl-0015.cnf" : 3012964503482414330783936367006634039953704207876657607,
                "sat-grid-pbl-0020.cnf" : 505529009203800755681036389586231130590495376930061598744188591803177007627164988944437560726719,
                "sat-grid-pbl-0025.cnf" : 18051755963842913831469740173876709710679368179331841451265906666609267268260641283484985491991777996404498069889539875259775880457350865091313349862141,
                "sat-grid-pbl-0030.cnf" : 154089430409193243326541334620745040441316292191902104056995281410078040886572580007404908279631836580121336881940079952338956656599622521005239507469477568002534440349139077306892061020210022834318422387583588123648727,
                "c432.isc" : 68719476736,
                "c499.isc" : 2199023255552,
                "c1355.isc" : 2199023255552,
                "c1908.isc" : 8589934592,
                "c2670.isc" : 13803492693581127574869511724554050904902217944340773110325048447598592,
        #        "c7552.isc" : 205688069665150755269371147819668813122841983204197482918576128
            }

            #same as model_counts dictionary but 2 lists for iterating in specified order, want to compute for c880 first due to assertion error
            model_names = ["c880.isc", "log-1.cnf" , "log-2.cnf" , "log-3.cnf" , "log-4.cnf" , "tire-1.cnf", "tire-2.cnf", "tire-3.cnf", "tire-4.cnf", "ra.cnf" , "rb.cnf" , "sat-grid-pbl-0010.cnf" , "sat-grid-pbl-0015.cnf" , "sat-grid-pbl-0020.cnf" , "sat-grid-pbl-0025.cnf" , "sat-grid-pbl-0030.cnf" , "c432.isc" , "c499.isc" , "c1355.isc" , "c1908.isc" , "c2670.isc"]
            model_counts_list = [1152921504606846976, 564153552511417968750, 32334741710, 279857462060, 23421510324076617565622131248, 726440820, 738969640920, 222560409176, 103191650628000, 18739277038847939886754019920358123424308469030992781557966909983211910963157763678726120154469030856807730587971859910379069462105489708001873004723798633342340521799560185957916958401869207109443355859123561156747098129524433371596461424856004227854241384374972430825095073282950873641, 538812462750928282721716308734898413194103864553832956073815148020987917365241105816807625188941823391012398976196112157887068449390989186368113511090801920577156367304804512491926215360520651047719401241944845883406098779319363314639309655779343140788696509960447536163784266937815828202118895534508004061478961506257883130142920912100543747226035966976598909666696626176, 593962746002226256572855, 3012964503482414330783936367006634039953704207876657607, 505529009203800755681036389586231130590495376930061598744188591803177007627164988944437560726719, 18051755963842913831469740173876709710679368179331841451265906666609267268260641283484985491991777996404498069889539875259775880457350865091313349862141, 154089430409193243326541334620745040441316292191902104056995281410078040886572580007404908279631836580121336881940079952338956656599622521005239507469477568002534440349139077306892061020210022834318422387583588123648727, 68719476736, 2199023255552, 2199023255552, 8589934592, 13803492693581127574869511724554050904902217944340773110325048447598592]
            assert(len(model_names) == len(model_counts_list))
#            model_names = ["rb.cnf"]
#            model_counts_list = [538812462750928282721716308734898413194103864553832956073815148020987917365241105816807625188941823391012398976196112157887068449390989186368113511090801920577156367304804512491926215360520651047719401241944845883406098779319363314639309655779343140788696509960447536163784266937815828202118895534508004061478961506257883130142920912100543747226035966976598909666696626176,]
    if RUN_SPECIAL:
        model_counts = {
            "lang12.cnf" : 10,
#            "wff.3.150.525.cnf" : np.exp(32.57),

        }

    exact_log_Zs = []
    our_estimators = []
    our_lower_bounds = []
    our_upper_bounds = []
    gumbel_upper_bounds = []

    track_progress = 0
    #save our results in a dictionary with-
    #key: string, model_txt_file
    #value: dictionary with-
    #   key: string, the estimator 'exact_log_Z', 'barv_estimator', 'barv_upper_bound', 'barv_lower_bound', 'gumbel_upper'
    #   value: list of floats, the value of the estimator in each experiment (list has length NUM_EXPERIMENTS)
    results = {}

#    for model_txt_file, Z in model_counts.iteritems():
    for model_txt_file, Z in zip(model_names, model_counts_list):
        for exp_num in range(NUM_EXPERIMENTS):
            print "starting on model:", track_progress, " name:", model_txt_file, 'experiment number:', exp_num
            sat_problem = read_SAT_problem("../SAT_problems/SAT_problems_cnf/%s" % model_txt_file)
            estimators = estimate_sharp_sat(sat_problem, gumbel_trials=NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, k=OUR_K, \
                chunk_size=CHUNK_SIZE, run_our_method=True, run_gumbel=False, run_new_gumbel_LB=True,
                log_base=LOG_BASE, time_limit=TIME_LIMIT)


    #        exact_log_Zs.append(Decimal(Z).ln())
    #        our_estimators.append(estimators['barv_estimator'])
    #        our_lower_bounds.append(estimators['barv_lower_bound'])
    #        our_upper_bounds.append(estimators['barv_upper_bound'])
    #        gumbel_upper_bounds.append(estimators['gumbel_upper'])

            estimators['exact_log_Z'] = [Decimal(Z).ln()/Decimal(LOG_BASE).ln()]
            if not model_txt_file in results:
                results[model_txt_file] = estimators
            else:
                for estimator_name, estimator_value in estimators.iteritems():
                    results[model_txt_file][estimator_name].append(estimator_value[0])

            #print "log(z) =", Decimal(Z).ln()#np.log(Z_sgp0010)
            #print "our estimator =", estimators['barv_estimator']
            #print "gumbel_upper bound =", estimators['gumbel_upper']

            track_progress+=1

####### skip plotting 
#######    fig = plt.figure()
#######    ax = plt.subplot(111)

########    ax.plot(range(len(model_counts)), our_estimators, 'bx', label='our estimator, k=%d' %OUR_K, markersize=10)
########    ax.plot(range(len(model_counts)), our_upper_bounds, 'b+', label='our upper bound, k=%d' %OUR_K, markersize=10)
########    ax.plot(range(len(model_counts)), our_lower_bounds, 'b^', label='our lower bound, k=%d' %OUR_K, markersize=5)
########    ax.plot(range(len(model_counts)), gumbel_upper_bounds, 'r+', label='gumbel upper bound num_perturb=%d' % NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, markersize=10)
########
#######
#######    plt.title('#SAT Model Count upper Bounds and Estimates')
#######    plt.xlabel('model index, not meaningful')
#######    plt.ylabel('log(Z)')
########    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#######
#######    #make the font bigger
#######    matplotlib.rcParams.update({'font.size': 15})
#######
#######    # Shrink current axis's height by 10% on the bottom
#######    box = ax.get_position()
#######    ax.set_position([box.x0, box.y0 + box.height * 0.1,
#######                     box.width, box.height * 0.9])
#######    
#######    # Put a legend below current axis
#######    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
#######              fancybox=False, shadow=False, ncol=2, numpoints = 1)
#######
#######
#######    fig.savefig('sat_estimators_k=%d_gumbelPerturbCount=%d_chunk_size=%d' % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, CHUNK_SIZE), bbox_extra_artists=(lgd,), bbox_inches='tight')    
#######    plt.close()   


    if RUN_SPECIAL:
        if CHUNK_SIZE:
            data_filename = "/atlas/u/jkuck/AAAI_2017/refactored_multi_model/results_data/lang12_sat_results_k=%d_gumbelPerturbCount=%d_chunk_size=%dnumExp=%d" % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, CHUNK_SIZE, NUM_EXPERIMENTS)
        else:
            data_filename = "/atlas/u/jkuck/AAAI_2017/refactored_multi_model/results_data/lang12_sat_results_k=%d_gumbelPerturbCount=%d_chunk_size=NonenumExp=%d" % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, NUM_EXPERIMENTS)


    else:
        if CHUNK_SIZE:
            data_filename = "/atlas/u/jkuck/AAAI_2017/refactored_multi_model/results_data/sat_results_k=%d_gumbelPerturbCount=%d_chunk_size=%dnumExp=%d" % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, CHUNK_SIZE, NUM_EXPERIMENTS)
        else:
            data_filename = "/atlas/u/jkuck/AAAI_2017/refactored_multi_model/results_data/sat_results_k=%d_gumbelPerturbCount=%d_chunk_size=NonenumExp=%d" % (OUR_K, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, NUM_EXPERIMENTS)

    save_results_data(data_filename, results, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, OUR_K, log_base=LOG_BASE)
    print_results_latex_table_ourGumbelPerturbLB(results, NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, OUR_K, LOG_BASE)





