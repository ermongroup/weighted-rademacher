from __future__ import division
import numpy as np
import maxflow
import matplotlib.pyplot as plt
import time

from pgmpy.models import MarkovModel

from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor
from itertools import product






def upper_bound_Z_gumbel(sg_model, num_trials):
    '''
    Upper bound the partition function of the specified spin glass model using
    an empircal estimate of the expected maximum energy perturbed with 
    Gumbel noise

    Inputs:
    - sg_model: type SG_model, specifies the spin glass model
    - num_trials: int, estimate the expected perturbed maximum energy as
        the mean over num_trials perturbed maximum energies

    Outputs:
    - upper_bound: type float, upper bound on ln(partition function)
    '''

    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])

    upper_bound = 0.0
    for i in range(num_trials):
        perturb_ones = np.random.gumbel(loc=-.5772, scale=1.0, size=(N,N))
        perturb_zeros = np.random.gumbel(loc=-.5772, scale=1.0, size=(N,N))
        map_state = spin_glass_perturbed_MAP_state(sg_model, perturb_ones, perturb_zeros)
        cur_perturbed_energy = calc_perturbed_energy(sg_model, perturb_ones, perturb_zeros, map_state)

#####        ########### DEBUGGING ###########
#####        (delta_exp, map_state_debug) = find_MAP_spin_glass(sg_model, perturb_ones, perturb_zeros)
#####        check_delta = np.log(delta_exp)
######        if cur_perturbed_energy != check_delta:
######            print "local field params:"
######            print sg_model.lcl_fld_params
######            print "cpl_params_h"
######            print sg_model.cpl_params_h
######            print "cpl_params_v"
######            print sg_model.cpl_params_v
######
######            print "perturb 1s"
######            print perturb_ones
######
######            print "perturb 0s"
######            print perturb_zeros
#####
#####        assert(np.abs(cur_perturbed_energy - check_delta) < .0001), (cur_perturbed_energy, check_delta, map_state, map_state_debug)
#####        cur_perturbed_energy = check_delta
#####        ########### END DEBUGGING ###########


        upper_bound += cur_perturbed_energy

#    upper_bound = upper_bound/num_trials
    upper_bound = upper_bound/num_trials
    return upper_bound

def estimate_Z_iid_gumbel(sg_model, num_trials):
    '''
    For debugging, generate iid gumbel for every state in the spin glass model
    to estimate the partition function and also compute log(Z) exactly
    '''
    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])
    all_states = [[0, 1] for i in range(N**2)]
    energies = np.zeros(2**(N**2))
    #- state: binary (0,1) array with dimensions (NxN), we are finding the perturbed
    #energy of this particular state. state[i,j] = 0 means node ij takes the value -1,
    #state[i,j] = 1 means node ij takes the value 1
    idx = 0
    print sg_model.lcl_fld_params
    for cur_state in product(*all_states):
        cur_state_array = np.zeros((N,N))
        for r in range(N):
            for c in range(N):
                cur_state_array[r,c] = cur_state[r*N + c]
        cur_energy = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), cur_state_array)

        energies[idx] = cur_energy
        
#        print '-'*30
#        print "idx =", idx
#        print "cur_energy", cur_energy
#        print "energies", energies

        idx += 1

    assert(idx == 2**(N**2))

    exact_log_Z = 0.0
    for cur_energy in energies:
#        print "cur_energy", cur_energy
        exact_log_Z += np.exp(cur_energy)
    exact_log_Z = np.log(exact_log_Z)
#   print exact_log_Z
    lfp = sg_model.lcl_fld_params[0]
#    assert(exact_log_Z == np.log(np.exp(lfp) + np.exp(-lfp))), (exact_log_Z, np.log(np.exp(lfp) + np.exp(-lfp)))
#    assert(exact_log_Z < 1.127 and exact_log_Z > .693), (exact_log_Z, sg_model.lcl_fld_params)
    log_Z_estimates = []
    for itr in range(num_trials):
        cur_gumbel_perturbations = np.random.gumbel(loc=0.0, scale=1.0, size=2**(N**2))
        cur_log_Z_est = np.max(energies+cur_gumbel_perturbations) - 0.5772
        log_Z_estimates.append(cur_log_Z_est)
    return (np.mean(log_Z_estimates), exact_log_Z)


def lower_bound_Z_gumbel(sg_model, copy_factor):
    '''
    Upper bound the partition function of the specified spin glass model using
    an empircal estimate of the expected maximum energy perturbed with 
    Gumbel noise

    Inputs:
    - sg_model: type SG_model, specifies the spin glass model
    - copy_factor: int, specifies the number of copies of each variable in the
        expanded model.  The original (N x N) spin glass model has been expanded
        to a model with dimensions (copy_factor*N x copy_factor*N), with copy_factor^2
        copies of each variable 

    Outputs:
    - lower_bound: type float, lower bound on ln(partition function)
    '''

    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])


    perturb_ones = np.random.gumbel(loc=-.5772, scale=1.0, size=(copy_factor*N, copy_factor*N))
    perturb_zeros = np.random.gumbel(loc=-.5772, scale=1.0, size=(copy_factor*N, copy_factor*N))

    expanded_map_state = expanded_spin_glass_perturbed_MAP_state(sg_model, copy_factor, perturb_ones, perturb_zeros)
    lower_bound = calc_expanded_perturbed_energy(sg_model, copy_factor, perturb_ones, perturb_zeros, expanded_map_state)

    return lower_bound



def upper_bound_Z_barvinok_k2(sg_model, k_2):
    '''
    Upper bound the partition function of the specified spin glass model using
    vector perturbations (uniform random in {-1,1}^n) inspired by Barvinok

    Inputs:
    - sg_model: type SG_model, specifies the spin glass model
    - k_2: int, take the mean of k_2 solutions to independently perturbed optimization
        problems to tighten the slack term between max and expected max

    Outputs:
    - upper_bound: type float, upper bound on ln(partition function)
    '''

    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])

    #we will take the mean of k_2 solutions to independently perturbed optimization
    #problems to tighten the slack term between max and expected max
    deltas = []
    for i in range(k_2):
        random_vec = np.random.rand(N,N)
        for r in range(N):
            for c in range(N):
                if random_vec[r,c] < .5:
                    random_vec[r,c] = -1
                else:
                    random_vec[r,c] = 1


        perturb_ones = random_vec
        perturb_zeros = -1*random_vec
        map_state = spin_glass_perturbed_MAP_state(sg_model, perturb_ones, perturb_zeros)
        cur_delta = calc_perturbed_energy(sg_model, perturb_ones, perturb_zeros, map_state)
        deltas.append(cur_delta)

    delta_bar = np.mean(deltas)

#############    ########### DEBUGGING ###########
#############    (delta_exp, map_state_debug) = find_MAP_spin_glass(sg_model, perturb_ones, perturb_zeros)
#############    check_delta = np.log(delta_exp)
#############    assert(np.abs(check_delta - delta_bar) < .0001)
#############
#############    #assert(cur_perturbed_energy == check_delta), (cur_perturbed_energy, check_delta)
#############    delta_bar = check_delta
#############    ########### END DEBUGGING ###########


    #option 1 for upper bound on log(permanent)
    total_num_random_vals = N**2
    upper_bound1 = delta_bar + np.sqrt(6*total_num_random_vals/k_2) + total_num_random_vals*np.log(1+1/np.e)




    #check optimize beta
    unperturbed_map_state = spin_glass_perturbed_MAP_state(sg_model, np.zeros((N,N)), np.zeros((N,N)))
    log_w_max = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), unperturbed_map_state)
    a_min = delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_max

    (w_min, w_min_state) = find_w_min_spin_glass(sg_model)
    log_w_min = np.log(w_min)
    a_max = delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_min


    def upper_bound_Z(delta_bar, n, log_w, log_w_type, k_2):
        '''
        Inputs:
        - delta_bar: float, our estimator of the log of the permanent
        - n: int, the total number of random values (c is in {-1,1}^n)
        - log_w: float, either log_w_min or log_w_max (log of the smallest or largest weight)
        - log_w_type: string, either "min" or "max" meaning we are upper bounding using
            either the largest or smallest weight
        - k_2: int, take the mean of k_2 solutions to independently perturbed optimization
            problems to tighten the slack term between max and expected max
        

        Ouputs:
        - log_Z_upper_bound: float, upper bound on log(Z)
        - beta: float, the optimized value of beta (see our writeup, ratio used in rules)
        '''        
        assert(log_w_type in ["min", "max"])
        a = delta_bar + np.sqrt(6*n/k_2) - log_w
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

        PRINT_DEBUG = True
        if PRINT_DEBUG:
            print "beta =", beta                            

        log_Z_upper_bound = np.log((1-beta)/beta)*a - n*np.log(1-beta) + log_w
        return log_Z_upper_bound, beta

    (upper_bound_w_min, beta_w_min) = upper_bound_Z(delta_bar, total_num_random_vals, log_w_min, "min", k_2)
    (upper_bound_w_max, beta_w_max) = upper_bound_Z(delta_bar, total_num_random_vals, log_w_max, "max", k_2)
    if np.isnan(upper_bound_w_min):
        upper_bound_w_min = np.PINF
    if np.isnan(upper_bound_w_max):
        upper_bound_w_max = np.PINF


    upper_bound_opt_beta = min([upper_bound_w_min, upper_bound_w_max, upper_bound1])
############    upper_bound_opt_beta = upper_bound1
############    upper_bound_opt_beta = min([upper_bound_w_max, upper_bound1])


#    if (delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_max)/total_num_random_vals >= .5 or \
#       (delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_max)/total_num_random_vals <= 1/(1+np.e) or \
#       (delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_min)/total_num_random_vals >= 1/(1+np.e) or \
#       (delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_min)/total_num_random_vals <= 0:
#       assert(upper_bound_opt_beta == upper_bound1), (upper_bound_opt_beta, [upper_bound_w_min, upper_bound_w_max, upper_bound1])
#    else:
#        assert(upper_bound_opt_beta == upper_bound_w_max or upper_bound_opt_beta == upper_bound_w_min)
#








    #corrected rademacher lower bound
    lambda_prime_w_max = (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_max)/total_num_random_vals
    lambda_prime_w_min = (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_min)/total_num_random_vals
    print "lambda_prime_w_max =", lambda_prime_w_max
    print "lambda_prime_w_min =", lambda_prime_w_min
    #check whether can we use w_max?
    if lambda_prime_w_max >= 1:
        print ":) :) we can use w_max for corrected rademacher!!"
        print "lambda =", lambda_prime_w_max
        corrected_rademacher_LB = (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_max)**2/(2*total_num_random_vals) + log_w_max
    #check whether can we use w_min?
    elif lambda_prime_w_min <= 1:
        print ":) we can use w_min for corrected rademacher!!"
        print "lambda =", lambda_prime_w_min
        print "delta_bar - np.sqrt(6*total_num_random_vals/k_2) =", delta_bar - np.sqrt(6*total_num_random_vals/k_2)
        print "log_w_min =", log_w_min
        print "2*total_num_random_vals =", 2*total_num_random_vals
        print "(delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_min)**2/(2*total_num_random_vals) =", (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_min)**2/(2*total_num_random_vals)
        corrected_rademacher_LB = (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_min)**2/(2*total_num_random_vals) + log_w_min
    else:
        print ":( we can't use w_min or w_max for corrected rademacher"
        corrected_rademacher_LB = delta_bar - np.sqrt(6*total_num_random_vals/k_2) - total_num_random_vals/2







    return (corrected_rademacher_LB, upper_bound_opt_beta)
############    return(upper_bound1, upper_bound_w_max, upper_bound_opt_beta)


def generate_random_fully_connected_MRF(N):
    '''
    Create an MRF with N binary nodes that are all part of the same clique, which
    has 2^N random values

    '''

    G = MarkovModel()

    #create an N nodes
    node_names = ['x%d' % i for i in range(N)]

    G.add_nodes_from(node_names)

    #add an edge between each node and its 4 neighbors, except when the
    #node is on the grid border and has fewer than 4 neighbors
    edges = []

    for i in range(N):
        for j in range(i+1, N):
            edges.append(('x%d' % i, 'x%d' % j))

    assert(len(edges) == .5*N**2)

    G.add_edges_from(edges) 


    single_factor = DiscreteFactor(edges, cardinality=[2 for i in range(N)], values=np.random.rand(*[2 for i in range(N)]))
    G.add_factors(single_factor)
 
    return G





def find_MAP_val(G):
    '''
    Inputs:
    - G: MarkovModel
    '''

    bp = BeliefPropagation(G)
    bp.max_calibrate()
    clique_beliefs = bp.get_clique_beliefs()
    map_val = np.max(clique_beliefs.values()[0].values)
    return map_val


def find_MAP_state(G):
    '''
    Inputs:
    - G: MarkovModel
    '''

    bp = BeliefPropagation(G)
    bp.max_calibrate()
    clique_beliefs = bp.get_clique_beliefs()
    phi_query = bp._query(G.nodes(), operation='maximize')
#    print phi_query
    return phi_query


def get_partition_function_BP(G):
    '''
    Calculate partition function of G using belief propogation

    '''
    bp = BeliefPropagation(G)
    bp.calibrate()
    clique_beliefs = bp.get_clique_beliefs()
    partition_function = np.sum(clique_beliefs.values()[0].values)
    return partition_function


def compare_upper_bounds(N, f, num_gumbel_trials, k_2, gumbel_LB_copy_factor, sg_neighbor_count):
    print "gumbel_LB_copy_factor =", gumbel_LB_copy_factor
    coupling_strengths = [i/10.0 for i in range(1, 30)]
#    coupling_strengths = [i/10.0 for i in range(12, 13)]
#    coupling_strengths = [i/100.0 for i in range(900, 1000)]
#    coupling_strengths = [1]
    gumbel_UBs = []
    gumbel_errors = []
    gumbel_LBs = []
    barvinok_UBs = []
    barvinok_LBs = []
    barvinok_errors = []
    barvinok_UB1s = []
    barvinok_UBwmaxs = []
    exact_log_Zs = [] #exact values of log(partition function)
    exact_log_Z2s = [] #debug, another list of exact values of log(partition function)

    iid_gumbel_estimates = []

    gumbel_barvinok_diff = []

    wishful_barvinok_LBs = []
    wishful_barvinok_UBs = []

    for (idx, c) in enumerate(coupling_strengths):
        print idx/len(coupling_strengths), 'fraction of the way through'
        sg_model = SG_model(N, f, c, neighbor_count=sg_neighbor_count)
        gumbel_UB = upper_bound_Z_gumbel(sg_model, num_gumbel_trials)
        gumbel_UBs.append(gumbel_UB) 
        CALC_GUMBEL_LB = False
        if CALC_GUMBEL_LB:       
            gumbel_LB = lower_bound_Z_gumbel(sg_model, gumbel_LB_copy_factor)
            gumbel_LBs.append(gumbel_LB)
        (barvinok_LB, barvinok_UB) = upper_bound_Z_barvinok_k2(sg_model, k_2=k_2)
        barvinok_UBs.append(barvinok_UB)
        barvinok_LBs.append(barvinok_LB)

        gumbel_barvinok_diff.append(gumbel_UB - barvinok_UB)

#        (iid_gumbel_estimate, exact_log_Z2) = estimate_Z_iid_gumbel(sg_model, 1000)
#        iid_gumbel_estimates.append(iid_gumbel_estimate)
#        exact_log_Z2s.append(exact_log_Z2)


#        (barvinok_UB1, barvinok_UB_w_max, barvinok_UB_min) = upper_bound_Z_barvinok(sg_model)
#        barvinok_UB1s.append(barvinok_UB1)
#        barvinok_UBwmaxs.append(barvinok_UB_w_max)
#        barvinok_UBs.append(barvinok_UB_min)

        CALC_EXACT_Z = True
        if CALC_EXACT_Z:    
            partition_function = eval_partition_func_spin_glass(sg_model)
            exact_log_Zs.append(np.log(partition_function))
    
            gumbel_errors.append(gumbel_UB - np.log(partition_function))
            barvinok_errors.append(barvinok_UB - np.log(partition_function))


        TEST_WISHFUL_BARVINOK = False
        if TEST_WISHFUL_BARVINOK:
            (wishful_barvinok_LB, wishful_barvinok_UB) = bound_Z_barvinok_wishful(sg_model, k_2=k_2)
            wishful_barvinok_LBs.append(wishful_barvinok_LB)
            wishful_barvinok_UBs.append(wishful_barvinok_UB)

    fig = plt.figure()
    ax = plt.subplot(111)


#    ax.plot(coupling_strengths, barvinok_UBs, 'bx', label='our upper bound min')
#    ax.plot(coupling_strengths, barvinok_UB1s, 'yx', label='our upper bound1')
#    ax.plot(coupling_strengths, barvinok_UBwmaxs, 'gx', label='our upper bound wmax')

    #plot our estimate, gumbel estimate, and exact log(Z)
    ax.plot(coupling_strengths, barvinok_UBs, 'bx', label='our upper bound')
    ax.plot(coupling_strengths, barvinok_LBs, 'b^', label='our lower bound')
    ax.plot(coupling_strengths, gumbel_UBs, 'rx', label='gumbel upper bound')
    if CALC_GUMBEL_LB:
        ax.plot(coupling_strengths, gumbel_LBs, 'r^', label='gumbel lower bound')
    if CALC_EXACT_Z:
        ax.plot(coupling_strengths, exact_log_Zs, 'g*', label='exact log(Z)')
    if TEST_WISHFUL_BARVINOK:
        ax.plot(coupling_strengths, wishful_barvinok_LBs, 'y^', label='barvinok wishful lower bound')
        ax.plot(coupling_strengths, wishful_barvinok_UBs, 'yx', label='barvinok wishful upper bound')

#    ax.plot(coupling_strengths, iid_gumbel_estimates, 'r+', label='iid gumbel estimate')
#    ax.plot(coupling_strengths, exact_log_Z2s, 'gx', label='exact log(Z2)')

    #plot our estimation error and gumbel estimation error
#    ax.plot(coupling_strengths, barvinok_errors, 'bx', label='our error')
#    ax.plot(coupling_strengths, gumbel_errors, 'rx', label='gumbel error')

   #plot difference (gumbel estimate - our estimate)
#    ax.plot(coupling_strengths, gumbel_barvinok_diff, 'bx', label='gumbel_UB - our_UB')

    
    plt.title('%dx%d model, f=%f, gumbel_trials=%d, our k=%d' % (N, N, f, num_gumbel_trials, k_2))
    plt.xlabel('coupling_strengths')
    plt.ylabel('ln(Z)')
    #plt.legend()
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('mrf_CUR_TEST_upper_bounds', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()    # close the figure
#    plt.show()



##########    fig = plt.figure()
##########    ax = plt.subplot(111)
##########
##########    #plot our estimate, gumbel estimate, and exact log(Z)
##########    ax.plot(coupling_strengths, barvinok_UBs, 'bx', label='our upper bound')
##########    ax.plot(coupling_strengths, gumbel_UBs, 'rx', label='gumbel upper bound')
##########    ax.plot(coupling_strengths, exact_log_Zs, 'g*', label='exact log(Z)')
##########
##########
##########
##########    plt.title('ln(Z) upper bounds')
##########    plt.xlabel('coupling_strengths')
##########    plt.ylabel('ln(Z)')
##########    #plt.legend()
##########    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
##########    fig.savefig('CUR_TEST_upper_bounds1', bbox_extra_artists=(lgd,), bbox_inches='tight')    
##########    plt.close()    # close the figure


if __name__=="__main__":
#    for i in range(50):
#        test_find_w_min_spin_glass()


    compare_upper_bounds(N=5, f=1.0, num_gumbel_trials=100, k_2=1, gumbel_LB_copy_factor=10, sg_neighbor_count=4)

#    compare_upper_bounds(N=100, f=1.0, num_gumbel_trials=10, k_2=10, gumbel_LB_copy_factor=70)

#    compare_upper_bounds(100, 1.0, 10)

#    sg_model = SG_model(N=5, f=1.0, c=3.0)
#    print "Z =", eval_partition_func_spin_glass(sg_model)