'''
We perform a simple comparison of our {-1,1} pertubation with 
low dimensional gumbel perturbations.  Specifically, we compare
both estimators (WHAT IS THE GUMBEL ESTIMATOR, DOESN'T EXIST, RIGHT?)
and bounds of the permanent of a matrix with elements in [0, 1).  
To simplify the initial comparison we:

1. Set all gumbel M_i's to 1
2. Set our k_1 and k_2 to 1

We compare with the gumbel lower bound both setting epsilon so
that the bound holds with .95 probability and setting epsilon to
0 with the bound holding with probability -infinity (meaningless)

Gumbel paper: https://arxiv.org/pdf/1602.03571.pdf
'''

from __future__ import division

import numpy as np
import math
from pymatgen.optimization import linear_assignment
import scipy.optimize
import cProfile
import matplotlib
matplotlib.use('Agg') #prevent error running remotely
import matplotlib.pyplot as plt
from permanent import permanent as rysers_permanent
import scipy.stats.mstats
import cvxpy as cvx
from itertools import product
import cProfile
from decimal import Decimal

def find_max_cost(cost_matrix):
    '''
    Solve the assignment problem for the specified cost_matrix

    Inputs:
    - cost_matrix: numpy.ndarray, cost matrix

    Outputs:
    - max_cost: float, the maximum cost
    - max_cost_assignments: list of pairs representing indices of 1's in max cost permutation matrix
    '''
    size = cost_matrix.shape[0]
    lin_assign = linear_assignment.LinearAssignment(-1*cost_matrix) #multiply by -1 to find max cost
    solution = lin_assign.solution
    max_cost_assignments = zip([i for i in range(len(solution))], solution)
    max_cost = 0.0
    for (row,col) in max_cost_assignments:
        assert(row < size), (row, size, cost_matrix)
        assert(col < size), (col, size, cost_matrix)
        max_cost += cost_matrix[row][col]
    return (max_cost, max_cost_assignments)

def calc_permanent_rysers(matrix):
    '''
    Exactly calculate the permanent of the given matrix user Ryser's method (faster than calc_permanent)
    '''
    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    #this looks complicated because the method takes and returns a complex matrix,
    #we are only dealing with real matrices so set complex component to 0
    return np.real(rysers_permanent(1j*np.zeros((N,N)) + matrix))


def test_calc_permanent():
    for N in range(1,10):
        M = np.ones((N,N))
        test_perm = calc_permanent(M)
        print test_perm, math.factorial(N)
        assert(test_perm == math.factorial(N))


def compute_gumbel_lower_bound(matrix, probability=None):
    '''
    Lower bound the specified matrice's log(permanent) using gumbel pertubations
    with the specified probability.  If probability==None, set epsilon to 0.

    Inputs:
    - matrix: numpy.ndarray, matrix whose permanent we are lower bounding
    - probability: float, the probability with which our bound must hold, 
        If probability==None, set epsilon to 0

    Outputs:
    - lower_bound: float, the lower bound on log(permanent) 
    '''
    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]
    gumbel_perturbations = np.random.gumbel(loc=-.5772, scale=1.0, size=(N, N))
    perturbed_matrix = matrix + gumbel_perturbations
    (max_cost, max_cost_assignments) = find_max_cost(perturbed_matrix)
    if probability != None:
        assert(probability>=0 and probability<=1)
        epsilon = np.sqrt(np.pi**2/(6*(1 - probability))*(1 + (N**N - N)/(N-1)))
    else:
        epsilon = 0.0
    lower_bound = max_cost - epsilon*N
    return lower_bound

def compute_gumbel_upper_bound(matrix, num_perturbations):
    '''
    Upper bound the specified matrice's log(permanent) in expectation 
    using gumbel pertubations as in equation (35) of gumbel paper

    Inputs:
    - matrix: numpy.ndarray, matrix whose permanent we are lower bounding
    - num_perturbations: int, the number of perturbations to estimate
        expecation for the upper bound


    Outputs:
    - upper_bound: float, the upper bound on log(permanent) 
    '''
    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]
    upper_bound = 0.0
    for itr in range(num_perturbations):
        gumbel_perturbations = np.random.gumbel(loc=-.5772, scale=1.0, size=(N, N))
        perturbed_log_weights = np.log(matrix) + gumbel_perturbations
        (max_cost, max_cost_assignments) = find_max_cost(perturbed_log_weights)
        upper_bound += max_cost

    upper_bound = upper_bound/num_perturbations
    return upper_bound

def sample_plus_minus_1(dim1, dim2):
    vec = np.random.random((dim1, dim2))
    for i in range(vec.shape[0]):
        for j in range(vec.shape[1]):
            if vec[i][j] < .5:
                vec[i][j] = -1
            else:
                vec[i][j] = 1
    return vec

def get_binary_list_neg_orthant(num_to_binary, num_digits):
    '''
    convert num_to_binary to binary and, then change 0's to -1's
    and 1's to 0's, return as a list of digits

    Inputs:
    - num_to_binary: int, integer we are converting to binary
    - num_digits: int, digits in the returned number (length of list)

    Ouputs:
    - binary_list: list of -1's and 0's, num_to_binary in binary
        with 0's changed to -1's and 1's to 0's.
        each digit stored as an element in the list
    '''       
    binary_list = [int(x) for x in bin(num_to_binary)[2:]]
    while len(binary_list) < num_digits:
        binary_list.insert(0, 0) #not efficient, change if performance critical!
    for idx in range(len(binary_list)):
        if binary_list[idx] == 0:
            binary_list[idx] = -1
        else:
            assert(binary_list[idx] == 1)
            binary_list[idx] = 0
    return binary_list

def get_binary_list(num_to_binary, num_digits):
    '''
    convert num_to_binary to binary and return as a list of digits
    e.g. 8 as [1, 0, 0, 0]

    Inputs:
    - num_to_binary: int, integer we are converting to binary
    - num_digits: int, digits in the returned number (length of list)

    Ouputs:
    - binary_list: list of 0's and 1's, num_to_binary in binary with
        each digit stored as an element in the list
    '''    
    binary_list = [int(x) for x in bin(num_to_binary)[2:]]
    while len(binary_list) < num_digits:
        binary_list.insert(0, 0) #not efficient, change if performance critical!

            #might be faster, double check correctness           
#            bin_idx_str = bin(col_idx)[2:]
#            bin_idx2 = [0 if i < len(random_vec) - len(bin_idx_str) else int(bin_idx_str[i - (len(random_vec) - len(bin_idx_str))]) for i in range(len(random_vec))]
#            assert(bin_idx1 == bin_idx2), (bin_idx1, bin_idx2)

    return binary_list    



def compute_cost_matrix_row(N, debugged=True):
    '''
    Sample 1 row of the nxn cost matrix

    Inputs:
    - N: int, length of row in the cost matrix

    Ouputs:
    - row: numpy.ndarray of size (N,), row of the cost matrix where 
        each element represents a cost
    '''
    num_random_vals = int(math.ceil(np.log(N)/np.log(2)))
#    print "num_random_vals =", num_random_vals
#    num_random_vals = int(N)
    random_vec = sample_plus_minus_1(num_random_vals, 1) #sample random values of -1 or 1
    random_vec = random_vec.flatten() #turn into 1-d array
    row = []
    for col_idx in range(N):
#            bin_idx1 = get_binary_list_neg_orthant(col_idx, len(random_vec))
        bin_idx1 = get_binary_list(col_idx, len(random_vec))

        assert(len(bin_idx1) == len(random_vec)), (len(bin_idx1), len(random_vec))
        cost = 0
        for idx, bin_digit in enumerate(bin_idx1):
            if debugged:
                cost += 2*bin_digit*random_vec[idx] - random_vec[idx]
            else:
                cost += bin_digit*random_vec[idx] #wrong
        row.append(cost)
    row = np.asarray(row)
    return row


def compute_cost_matrix_row_more_randomness(N, debugged=True):
    '''
    Sample 1 row of the nxn cost matrix

    Inputs:
    - N: int, length of row in the cost matrix

    Ouputs:
    - row: numpy.ndarray of size (N,), row of the cost matrix where 
        each element represents a cost
    '''
#    num_random_vals = int(math.ceil(np.log(N)/np.log(2)))
    num_random_vals = int(N)
    random_vec = sample_plus_minus_1(num_random_vals, 1) #sample random values of -1 or 1
    random_vec = random_vec.flatten() #turn into 1-d array
    row = []

    random_mapings = []
    for col_idx in range(N):
        while True:
            cur_col_random_mapping = np.random.binomial(1, p=.5, size=N)
            if not(tuple(cur_col_random_mapping) in random_mapings):
                random_mapings.append(tuple(cur_col_random_mapping))
                break
    assert(len(random_mapings) == N)
                
    for col_idx in range(N):
#            bin_idx1 = get_binary_list_neg_orthant(col_idx, len(random_vec))
        bin_idx1 = random_mapings[col_idx]

        assert(len(bin_idx1) == len(random_vec)), (len(bin_idx1), len(random_vec))
        cost = 0
        for idx, bin_digit in enumerate(bin_idx1):
            if debugged:
                cost += 2*bin_digit*random_vec[idx] - random_vec[idx]
            else:
                cost += bin_digit*random_vec[idx] #wrong
        row.append(cost)
    row = np.asarray(row)
    return row

def compute_cost_matrix(N, debugged=True):
    '''
    Compute the nxn cost matrix as explained above in approx_permanent3 notes step 2.
    '''
    rows = []
    for i in range(N):
        cur_row = compute_cost_matrix_row(N, debugged)
#        cur_row = compute_cost_matrix_row_more_randomness(N, debugged)
        rows.append(cur_row)
    cost_matrix = np.asarray(rows)
    return cost_matrix

def sample_spherical(dim1, dim2):
    '''
    Sample a point from the unit sphere with dimension dim1*dim2,
    returning the point as a 2d array

    Inputs:
    - dim1: type int, dimension 1 of the array we return
    - dim2: type int, dimension 2 of the array we return

    Outputs:
    - vec: type numpy.ndarray (dim1 x dim2), the sampled point
    '''
    vec = np.random.randn(dim1, dim2)
    vec /= np.linalg.norm(vec)
    l2_norm = 0.0
    for i in range(vec.shape[0]):
        for j in range(vec.shape[1]):
            l2_norm += vec[i][j]**2
    assert(np.abs(l2_norm - 1.0) < .0000001)
    return vec

def approx_permanent3(matrix,  k_2, debugged=True, log_w_min=None):
    '''
    Approximate the permanent of the specified matrix using the gamma from 1.2.
    To accomplish this, we need to map up to N! permutations of our matrix to
    2^total_num_random_vals vectors in {-1,1}^total_num_random_vals for some 
    number total_num_random_vals.  We pick total_num_random_vals = N*log(N) so that
    2^(N*log(N)) = (2^log(N))^N = N^N > N!. 


    Steps:
    1. generate N*log(N) random values of -1 or 1 (each with probability .5),
    where log(N) values will be used to compute costs for each of the N rows in our matrix
    2. compute N^2 costs using the random numbers from (1), particularly using log(N)
    random numbers for each of the N rows
    3. solve the assignment problem using the costs from (2)

    *note, the code below doesn't exactly follow the steps in this order, but this is the idea*

    Inputs:
    - matrix: type numpy.ndarray of size NxN, the matrix whose permanent we are approximating,

    Outputs:
    - log_perm_estimate: float, log_perm_estimate for the permanent of the specified matrix (gamma*n), where
        n=total_num_random_vals=N*log(N)
    - lower_bound: float, lower bound to the permanent that holds with probability > .95
    - upper_bound: float, upper bound to the permanent that holds with probability > .95
    '''


    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    #we will generate a random vector in {-1,1}^total_num_random_vals
    total_num_random_vals = N*int(math.ceil(np.log(N)/np.log(2)))


    #we will take the mean of k_2 solutions to independently perturbed optimization
    #problems to tighten the slack term between max and expected max
    deltas = []
    for i in range(k_2):
        random_cost_matrix = compute_cost_matrix(N, debugged=debugged)
#        random_cost_matrix = sample_spherical(N,N)
        (cur_delta, max_cost_assignments) = find_max_cost(np.log(matrix) + random_cost_matrix)
#        print "cur_delta =", cur_delta
        deltas.append(cur_delta)


    delta_bar = np.mean(deltas)


#    print "delta_bar =", delta_bar
#    print "(delta_bar**2)*N*np.log(N) =", (delta_bar**2)*N*np.log(N)
#    print "N =", N
#    print "N! =", math.factorial(N)
#    print "ln(N!) =", np.log(math.factorial(N))
#    sleep(4)

    #option 1 for upper bound on log(permanent) (which is currently referred to as
    #the lower bound on gamma in the write up)
    if debugged:
        upper_bound1 = delta_bar + 7*np.sqrt(total_num_random_vals/k_2) + total_num_random_vals*np.log(1+1/np.e)
        #check taking out the slack
#        upper_bound1 = delta_bar + total_num_random_vals*np.log(1+1/np.e)
    else:
        upper_bound1 = delta_bar + 7*np.sqrt(total_num_random_vals/k_2) + total_num_random_vals*np.log(1+1/np.sqrt(np.e))

    #option 2 for upper bound on log(permanent) (which is currently referred to as
    #the lower bound on gamma in the write up)

    if log_w_min == None:
        (neg_min_log_weight, assignments_ignored) = find_max_cost(-np.log(matrix))
        log_w_min = -1 * neg_min_log_weight #log of the smallest weight
#    assert(log_w_min < 0.0)
    assert(log_w_min <= 0.0)
    upper_bound2 = (delta_bar + 7*np.sqrt(total_num_random_vals/k_2) - log_w_min)*np.log(total_num_random_vals-1) \
                 + 1.5 + log_w_min



    #check optimize beta
    (log_w_max, assignments_ignored) = find_max_cost(np.log(matrix))
    a_min = delta_bar + 7*np.sqrt(total_num_random_vals/k_2) - log_w_max
    a_max = delta_bar + 7*np.sqrt(total_num_random_vals/k_2) - log_w_min

    print "a_min/n =", a_min/total_num_random_vals
    print "a_max/n =", a_max/total_num_random_vals
####    if a_min/total_num_random_vals > 1/(1+np.e):
####        print "use a_min"
####        if a_min/total_num_random_vals >= 1:
####            beta = 1
####        else:
####            beta = a_min/total_num_random_vals
####        upper_bound_opt_beta = np.log((1-beta)/beta)*a_min - total_num_random_vals*np.log(1-beta) + np.log()
####
####    elif a_max/total_num_random_vals < 1/(1+np.e):
####        print "use a_max"
####        upper_bound_opt_beta = delta_bar + 7*np.sqrt(total_num_random_vals/k_2) + total_num_random_vals*np.log(1+1/np.e)
####
####    else:
####        print "use beta = 1/(1+e)"
####        upper_bound_opt_beta = delta_bar + 7*np.sqrt(total_num_random_vals/k_2) + total_num_random_vals*np.log(1+1/np.e)

    def upper_bound_Z(delta, n, log_w, log_w_type):
        '''
        Inputs:
        - delta: float, our estimator of the log of the permanent
        - n: int, the total number of random values (c is in {-1,1}^n)
        - log_w: float, either log_w_min or log_w_max (log of the smallest or largest weight)
        - log_w_type: string, either "min" or "max" meaning we are upper bounding using
            either the largest or smallest weight

        Ouputs:
        - log_Z_upper_bound: float, upper bound on log(Z)
        - beta: float, the optimized value of beta (see our writeup, ratio used in rules)
        '''        
        assert(log_w_type in ["min", "max"])
        a = delta + 7*np.sqrt(n/k_2) - log_w
        if log_w_type == "max":
            if a/n > .5:
                beta = .5
#            if a/n > 1:
#                beta = 1
            elif a/n < 1/(1+np.e):
                beta = 1/(1+np.e)
            else:
                beta = a/n
        elif log_w_type == "min":
            if a/n > 1/(1+np.e):
                beta = 1/(1+np.e)
            elif a/n < 0:
                beta = 0
            else:
                beta = a/n                
        log_Z_upper_bound = np.log((1-beta)/beta)*a - n*np.log(1-beta) + log_w
        return log_Z_upper_bound, beta

    def lower_bound_Z(delta, n, log_w, log_w_type):
        '''
        Inputs:
        - delta: float, our estimator of the log of the permanent
        - n: int, the total number of random values (c is in {-1,1}^n)
        - log_w: float, either log_w_min or log_w_max (log of the smallest or largest weight)
        - log_w_type: string, either "min" or "max" meaning we are upper bounding using
            either the largest or smallest weight

        Ouputs:
        - log_Z_upper_bound: float, upper bound on log(Z)
        - beta: float, the optimized value of beta (see our writeup, ratio used in rules)
        '''        
        assert(log_w_type in ["min"])
        a = delta + 7*np.sqrt(n/k_2) - log_w
        if log_w_type == "min":
            if a/n > .99:
                beta = .99
            elif a/n < .51:
                beta = .51
            else:
                beta = a/n
            
        print "a/n =", a/n
        log_Z_upper_bound = np.log((1-beta)/beta)*a - n*np.log(1-beta) + log_w
        return log_Z_upper_bound, beta

    print 'hi5'
    (upper_bound_w_min, beta_w_min) = upper_bound_Z(delta_bar, total_num_random_vals, log_w_min, "min")
    (upper_bound_w_max, beta_w_max) = upper_bound_Z(delta_bar, total_num_random_vals, log_w_max, "max")
    if np.isnan(upper_bound_w_min):
        upper_bound_w_min = np.PINF
    if np.isnan(upper_bound_w_max):
        upper_bound_w_max = np.PINF
    print 'hi6'
    print upper_bound_w_min, upper_bound_w_max, upper_bound1

    upper_bound_opt_beta = min([upper_bound_w_min, upper_bound_w_max, upper_bound1])
#    if upper_bound_w_min <= upper_bound1 and upper_bound_w_min <= upper_bound_w_max:
#        upper_bound_opt_beta = upper_bound_w_min
#        print "using w_min when optimizing beta, beta =", beta_w_min, "delta_bar =", delta_bar, "n =", total_num_random_vals, "log_w =", log_w_min, "upper_bound_opt_beta =", upper_bound_opt_beta, "upper_bound1 =", upper_bound1, "exact log Z =", np.log(calc_permanent_rysers(matrix))
#    elif upper_bound_w_max <= upper_bound1 and upper_bound_w_max < upper_bound_w_min:
#        upper_bound_opt_beta = upper_bound_w_max
#        print "using w_max when optimizing beta, beta =", beta_w_max, "delta_bar =", delta_bar, "n =", total_num_random_vals, "log_w =", log_w_max, "upper_bound_opt_beta =", upper_bound_opt_beta, "upper_bound1 =", upper_bound1, "exact log Z =", np.log(calc_permanent_rysers(matrix))
#    else:
#        assert(upper_bound1 < upper_bound_w_max and upper_bound1 < upper_bound_w_min), (upper_bound1, upper_bound_w_min, upper_bound_w_max)
#        upper_bound_opt_beta = upper_bound1
#        print "using upper_bound1 when optimizing beta, beta = 1/(1+e)"
#    print 'hi7'

    #corrected rademacher lower bound
    lambda_prime_w_max = (delta_bar - log_w_max)/total_num_random_vals
    lambda_prime_w_min = (delta_bar - log_w_min)/total_num_random_vals
    #check whether can we use w_max?
    if lambda_prime_w_max >= 1:
        print ":) :) we can use w_max for corrected rademacher!!"
        print "lambda =", lambda_prime_w_max
        corrected_rademacher_LB = (delta_bar - log_w_max)**2/(2*total_num_random_vals) + log_w_max
    #check whether can we use w_max?
    elif lambda_prime_w_min <= 1:
        print ":) we can use w_min for corrected rademacher!!"
        print "lambda =", lambda_prime_w_min
        print "delta_bar =", delta_bar
        print "log_w_min =", log_w_min
        print "2*total_num_random_vals =", 2*total_num_random_vals
        print "(delta_bar - log_w_min)**2/(2*total_num_random_vals) =", (delta_bar - log_w_min)**2/(2*total_num_random_vals)
        corrected_rademacher_LB = (delta_bar - log_w_min)**2/(2*total_num_random_vals) + log_w_min
    else:
        print ":( we can't use w_min or w_max for corrected rademacher"
        corrected_rademacher_LB = delta_bar - total_num_random_vals/2


    stefanos_new_copy_lower_bound = (delta_bar - log_w_min)**2/(2*(total_num_random_vals + log_w_max - log_w_min)) + log_w_min

    return (delta_bar, \
            upper_bound1, upper_bound2, upper_bound_opt_beta, \
            corrected_rademacher_LB, stefanos_new_copy_lower_bound)

def sample_random_matrix(dim1, dim2):
    vec = np.random.random((dim1, dim2))
    for i in range(vec.shape[0]):
        for j in range(vec.shape[1]):
            if vec[i][j] < .5:
                vec[i][j] = 0
#                vec[i][j] = .03
            else:
                vec[i][j] = 1
#                vec[i][j] = np.random.random()
#                vec[i][j] = .56
    return vec  


def create_diagonal(in_matrix, n):
    '''
    create diag_matrix, a diagonal matrix with n copies of in_matrix on it's diagonal 
    Inputs:
    - in_matrix: numpy array,
    - n: int, 
    - log_w: float, either log_w_min or log_w_max (log of the smallest or largest weight)
    - log_w_type: string, either "min" or "max" meaning we are upper bounding using
        either the largest or smallest weight

    Ouputs:
    - diag_matrix: numpy array, the array we created with shape of 
        in_matrix.shape[0]*n X in_matrix.shape[1]*n
    '''      
    diag_matrix = np.zeros((in_matrix.shape[0]*n, in_matrix.shape[1]*n))
    print diag_matrix.shape
    for i in range(n):
        diag_matrix[i*in_matrix.shape[0]:(i+1)*in_matrix.shape[0], \
                    i*in_matrix.shape[1]:(i+1)*in_matrix.shape[1]] = in_matrix
    return diag_matrix

if __name__ == "__main__":
##    np.random.seed(0)
##    test_improve_lower_bound()
##    sleep(4)


#    i = 3
#    N = 5

#    matrix = np.random.random((N, N))
#    sub_matrix_exact_permanent = calc_permanent_rysers(matrix)
#    matrix = create_diagonal(matrix, i+1)
#    exact_permanent = sub_matrix_exact_permanent**(i+1)
#    print sub_matrix_exact_permanent**(i+1)
#    print calc_permanent_rysers(matrix) 
#    assert(exact_permanent == calc_permanent_rysers(matrix))

#    sleep(23)

    NUM_TEST_MATRICES = 10
    N = 20
    #compute expectation of gumbel upper bound averaged over this many perturbations
    NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS = 10
    exact_log_permanents = []
    our_estimators = []
    our_estimators_log_base_corrected = []
    corrected_rademacher_LBs = []

    our_upper_bound1s = []
    our_upper_bound2s = []
    our_upper_bound_opt_betas = []
    gumbel_lower_bounds_epsilon0 = []
    gumbel_lower_bounds_95prob = []
    gumbel_upper_bounds = []
    rademacher_upper_bounds = []
    matrix_dimensions = [] #this is n in the writeup

    stefanos_copy_lower_bounds = []

    estimator_hard_upper_bounds = [] #trying to debug estimator, should not exceed this
    for i in range(NUM_TEST_MATRICES):
#    for i in range(5,NUM_TEST_MATRICES):
#    for i in range(1,5):
        print i
#        matrix = np.random.random((N, N))*(10**(-165))**(1/(N*(i+1)))
#        matrix = np.random.random((N, N))*(10**(-74))**(1/(N*(i+1)))
        matrix = np.random.random((N, N))
#        matrix = sample_random_matrix(N, N)
        estimator_hard_upper_bounds.append((N*(i+1))*np.log((N*(i+1)))/np.log(2))
#        matrix = np.ones((N, N))
#        print matrix
#        print (10**(-200))**(1/(N*(i+1)))
#        print (10**(-80))
#        print (1/(N*(i+1)))
#        sleep(4)

#        matrix = sample_random_matrix(N, N)*(1.0/1000000000000000000)**(1/(N*(i+1)))
        (neg_min_log_weight, assignments_ignored) = find_max_cost(-np.log(matrix))
        log_w_min = -1 * neg_min_log_weight #log of the smallest weight

        print "log_w_min =", (i+1)*log_w_min
        (TEMP_max_log_weight, assignments_ignored) = find_max_cost(np.log(matrix))
        print "TEMP_max_log_weight =", (i+1)*TEMP_max_log_weight


        MAKE_DIAGONAL_MATRIX = True
        EXPAND_NONDIAGONAL = False
        if MAKE_DIAGONAL_MATRIX:
            matrix_dimension = N*(i+1)

            sub_matrix_exact_permanent = calc_permanent_rysers(matrix)
            matrix = create_diagonal(matrix, i+1)
            exact_permanent = sub_matrix_exact_permanent**(i+1)
            exact_log_permanents.append(np.log(exact_permanent))
            diag_log_w_min = (i+1)*log_w_min

            print '@'*20
            print "exact_permanent =", exact_permanent
            print "log(exact_permanent) =", np.log(exact_permanent)
            (diag_log_w_max_FOR_PRINTING_ONLY, assignments_ignored1) = find_max_cost(np.log(matrix))
            #diag_log_w_max_FOR_PRINTING_ONLY = (i+1)*max_log_weight_FOR_PRINTING_ONLY
            print "n + log(w_max) roughly", (matrix_dimension)*np.log((matrix_dimension)) + diag_log_w_max_FOR_PRINTING_ONLY, "diag_log_w_max_FOR_PRINTING_ONLY =", diag_log_w_max_FOR_PRINTING_ONLY
            print "(n + log(w_min))/2 roughly", ((matrix_dimension)*np.log((matrix_dimension)) + diag_log_w_min)/2, "diag_log_w_min =", diag_log_w_min
            
            print "log(w_max) =", diag_log_w_max_FOR_PRINTING_ONLY
            print "log(w_min) =", diag_log_w_min
            print "matrix:"
            print matrix
            print '@'*20



        elif EXPAND_NONDIAGONAL:
            matrix_dimension = N*(i+1)

            #matrix = np.random.random((matrix_dimension, matrix_dimension))
            
            #matrix = sample_random_matrix(matrix_dimension, matrix_dimension)
            #exact_permanent = 1 #don't calculate
            #exact_log_permanents.append(np.log(exact_permanent))
            
            matrix = np.ones((matrix_dimension, matrix_dimension))
            exact_permanent = math.factorial(matrix_dimension)
            exact_log_permanents.append(Decimal(exact_permanent).ln())
            

            (neg_min_log_weight_expanded, assignments_ignored) = find_max_cost(-np.log(matrix))
            diag_log_w_min = -1 * neg_min_log_weight_expanded #log of the smallest weight


        else:
            exact_permanent = calc_permanent_rysers(matrix)
            exact_log_permanents.append(np.log(exact_permanent))
            matrix_dimension = N

        matrix_dimensions.append(matrix_dimension)


#        matrix = np.ones((N, N))
#        exact_permanent = math.factorial(N)
#        exact_log_permanents.append(np.log(float(exact_permanent)))
        

#        cProfile.run('approx_permanent3(matrix, debugged=True)')

        (log_perm_estimate, upper_bound1, \
            upper_bound2, upper_bound_opt_beta,\
            corrected_rademacher_LB, stefanos_new_copy_lower_bound) = \
            approx_permanent3(matrix, k_2=100, debugged=True, log_w_min=diag_log_w_min)
        print "a"
        #our_estimators.append(log_perm_estimate - N*(i+1))
        our_estimators.append(log_perm_estimate)
        our_estimators_log_base_corrected.append(log_perm_estimate*np.log(2))
        our_upper_bound1s.append(upper_bound1)
        our_upper_bound2s.append(upper_bound2)
        our_upper_bound_opt_betas.append(upper_bound_opt_beta)
        corrected_rademacher_LBs.append(corrected_rademacher_LB)

        stefanos_copy_lower_bounds.append(stefanos_new_copy_lower_bound)

        gumbel_lower_bound_epsilon0 = compute_gumbel_lower_bound(matrix)
        print "b"

        #gumbel_lower_bound_95prob = compute_gumbel_lower_bound(matrix, probability=.95)
        gumbel_lower_bound_95prob = gumbel_lower_bound_epsilon0
        print "c"

        gumbel_upper_bound = compute_gumbel_upper_bound(matrix, num_perturbations=NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS)
        print "d"

 
        gumbel_lower_bounds_epsilon0.append(gumbel_lower_bound_epsilon0)
        gumbel_lower_bounds_95prob.append(gumbel_lower_bound_95prob)
        gumbel_upper_bounds.append(gumbel_upper_bound)


    fig = plt.figure()
    ax = plt.subplot(111)

    if not EXPAND_NONDIAGONAL:
        ax.plot(matrix_dimensions, exact_log_permanents, 'go', label='exact permanents')
#    ax.plot(matrix_dimensions, our_estimators, 'r*', label='our estimator')
    ax.plot(matrix_dimensions, corrected_rademacher_LBs, 'b^', label='corrected_rademacher_LBs')
    ax.plot(matrix_dimensions, stefanos_copy_lower_bounds, 'g^', label="stefano's new copy lower bound")
    #ax.plot(matrix_dimensions, gumbel_lower_bounds_epsilon0, 'g^', label='gumbel lower bound, epsilon=0')
#    ax.plot(matrix_dimensions, lower_bound_w_max_3s, 'b^', label='lower_bound_w_max_3s')
#    ax.plot(matrix_dimensions, gumbel_lower_bounds_95prob, 'b^', label='gumbel lower bound, .95 probability')
#    plt.title('Permanent lower Bounds and Estimates, Gumbel .95 prob, %dx%d uniform random matrix in [0,1)' % (N, N))
    plt.title('Permanent lower Bounds and Estimates')
    plt.xlabel('matrix dimension (width and height)')
    plt.ylabel('log(permanent)')
    #plt.legend()
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.savefig('lower_bounds_gumbel_95prob.pdf')
#    plt.savefig('CUR_TEST_lower_bounds_%dx%d.pdf' % (N, N))
    fig.savefig('CUR_TEST_lower_bounds', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()    # close the figure
#    plt.show()



    fig = plt.figure()
    ax = plt.subplot(111)
#    if not EXPAND_NONDIAGONAL:
    if True:
        ax.plot(matrix_dimensions, exact_log_permanents, 'go', label='exact permanents')

    ax.plot(matrix_dimensions, our_estimators, 'r*', label='our estimator')
    ax.plot(matrix_dimensions, our_estimators_log_base_corrected, 'g*', label='our estimator log base corrected')
    ax.plot(matrix_dimensions, our_upper_bound1s, 'ys', label='our upper bound, beta option 1')
#    ax.plot(matrix_dimensions, our_upper_bound2s, 'rs', label='our upper bound, beta option 2')
    ax.plot(matrix_dimensions, our_upper_bound_opt_betas, 'gs', label='our upper bound, optimize beta')
    ax.plot(matrix_dimensions, gumbel_upper_bounds, 'bs', label='gumbel upper bound num_perturb=%d' % NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS)

    ax.plot(matrix_dimensions, estimator_hard_upper_bounds, 'vr', label='hard upper bound debug for our estimator')


    plt.title('Permanent upper Bounds and Estimates')
    plt.xlabel('matrix dimension (width and height)')
    plt.ylabel('log(permanent)')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('CUR_TEST_upper_bounds', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()   