from __future__ import division
import numpy as np
import maxflow
import matplotlib.pyplot as plt

from pgmpy.models import MarkovModel

from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor



################## Create a graph with integer capacities.
#################g = maxflow.Graph[int](2, 2)
################## Add two (non-terminal) nodes. Get the index to the first one.
#################nodes = g.add_nodes(2)
################## Create two edges (forwards and backwards) with the given capacities.
################## The indices of the nodes are always consecutive.
#################g.add_edge(nodes[0], nodes[1], 1, 2)
################## Set the capacities of the terminal edges...
################## ...for the first node.
#################g.add_tedge(nodes[0], -2, -5)
################## ...for the second node.
#################g.add_tedge(nodes[1], -9, 4)
#################
#################
#################flow = g.maxflow()
#################print "Maximum flow:", flow
#################
#################print "Segment of the node 0:", g.get_segment(nodes[0])
#################print "Segment of the node 1:", g.get_segment(nodes[1])
#################
#################sleep(4)


class SG_model:
    def __init__(self, N, f, c):
        '''
        Sample local field parameters and coupling parameters to define a spin glass model
        
        Inputs:
        - N: int, the model will be a grid with shape (NxN)
        - f: float, local field parameters (theta_i) will be drawn uniformly at random
            from [-f, f] for each node in the grid
        - c: float, coupling parameters (theta_ij) will be drawn uniformly at random from
            [0, c) (gumbel paper uses [0,c], but this shouldn't matter) for each edge in 
            the grid

        Values defining the spin glass model:
        - lcl_fld_params: array with dimensions (NxN), local field parameters (theta_i)
            that we sampled for each node in the grid 
        - cpl_params_h: array with dimensions (N x N-1), coupling parameters (theta_ij)
            for each horizontal edge in the grid.  cpl_params_h[k,l] corresponds to 
            theta_ij where i is the node indexed by (k,l) and j is the node indexed by
            (k,l+1)
        - cpl_params_v: array with dimensions (N-1 x N), coupling parameters (theta_ij)
            for each vertical edge in the grid.  cpl_params_h[k,l] corresponds to 
            theta_ij where i is the node indexed by (k,l) and j is the node indexed by
            (k+1,l)     
        '''
        self.N = N

        #sample local field parameters (theta_i) for each node
        self.lcl_fld_params = np.random.uniform(low=-f, high=f, size=(N,N))

        #sample horizontal coupling parameters (theta_ij) for each horizontal edge
        self.cpl_params_h = np.random.uniform(low=0.0, high=c, size=(N,N-1))

        #sample vertical coupling parameters (theta_ij) for each vertical edge
        self.cpl_params_v = np.random.uniform(low=0.0, high=c, size=(N-1,N))


def spin_glass_perturbed_MAP_state(sg_model, perturb_ones, perturb_zeros):
    '''
    Find the MAP state of a perturbed spin glass model using a min-cut/max-flow
    solver.  This method is faster than a MAP solver for a general MRF, but can
    only be used for special cases of MRF's.

    For info about what models can be used see:
    http://www.cs.cornell.edu/~rdz/Papers/KZ-PAMI04.pdf

    For info about transforming the MAP problem into a min-cut problem see:
    https://www.classes.cs.uchicago.edu/archive/2006/fall/35040-1/gps.pdf

    (N is an implicit parameter, the glass spin model has shape (NxN))


    Inputs:
    - sg_model: type SG_model, specifies the spin glass model   
    - perturb_ones: array with dimensions (NxN), specifies a perturbation of each
        node's potential when the node takes the value 1
    - perturb_zeros:  array with dimensions (NxN), specifies a perturbation of each
        node's potential when the node takes the value 0

    Outputs:
    - map_state: binary array with dimensions (NxN), the state in the spin glass model
        with the largest energy.  map_state[i,j] = 0 means node (i,j) is zero in the
        map state and map_state[i,j] = 1 means node (i,j) is one in the map state
    '''
    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])
    g = maxflow.GraphFloat()
    #Create (NxN) grid of nodes nodes
    nodes = g.add_nodes(N**2)

    #Add single node potentials
    for r in range(N):
        for c in range(N):
#            assert(np.abs(sg_model.lcl_fld_params[r,c]) + sg_model.lcl_fld_params[r,c] +\
#                                    np.abs(perturb_ones[r,c] - perturb_zeros[r,c]) >= 0)
#            assert(np.abs(sg_model.lcl_fld_params[r,c]) - sg_model.lcl_fld_params[r,c] +\
#                                    np.abs(perturb_ones[r,c] - perturb_zeros[r,c]) +\
#                                    perturb_ones[r,c] - perturb_zeros[r,c] >= 0), (np.abs(sg_model.lcl_fld_params[r,c]) - sg_model.lcl_fld_params[r,c] +\
#                                    np.abs(perturb_ones[r,c] - perturb_zeros[r,c]) +\
#                                    perturb_ones[r,c] - perturb_zeros[r,c])
#
#            perturbed_0_potential =2+ np.abs(sg_model.lcl_fld_params[r,c]) + sg_model.lcl_fld_params[r,c] +\
#                                    np.abs(perturb_ones[r,c] - perturb_zeros[r,c])
#            perturbed_1_potential =2+ np.abs(sg_model.lcl_fld_params[r,c]) - sg_model.lcl_fld_params[r,c] +\
#                                    np.abs(perturb_ones[r,c] - perturb_zeros[r,c]) +\
#                                    perturb_ones[r,c] - perturb_zeros[r,c]

            lamda_i = (sg_model.lcl_fld_params[r,c] + perturb_ones[r,c]) \
                    - (-sg_model.lcl_fld_params[r,c] + perturb_zeros[r,c])


            perturbed_0_potential = np.max((0, lamda_i))
            perturbed_1_potential = np.max((0, -lamda_i))

#            g.add_tedge(nodes[r*N + c], perturbed_0_potential, perturbed_1_potential)
            g.add_tedge(nodes[r*N + c], perturbed_1_potential, perturbed_0_potential)

    #Add two node potentials
    edge_count = 0
    for r in range(N):
        for c in range(N):
            #add a horizontal edge
            if c < N-1:
                g.add_edge(nodes[r*N + c], nodes[r*N + c+1], 2*sg_model.cpl_params_h[r,c], 2*sg_model.cpl_params_h[r,c])
                edge_count += 1
            #add a vertical edge
            if r < N-1:
                g.add_edge(nodes[r*N + c], nodes[(r+1)*N + c], 2*sg_model.cpl_params_v[r,c], 2*sg_model.cpl_params_v[r,c])
                edge_count += 1
    assert(edge_count == 2*N*(N-1))

    #find the maximum flow
    flow = g.maxflow()

    #read out node partitioning for max-flow
    map_state = np.zeros((N,N))
    for r in range(N):
        for c in range(N):
            map_state[r,c] = g.get_segment(nodes[r*N + c])
            assert(map_state[r,c] == 0 or map_state[r,c] == 1)

    return map_state

def calc_perturbed_energy(sg_model, perturb_ones, perturb_zeros, state):
    '''
    Compute the perturbed energy of the given state in a spin glass model
    (N is an implicit parameter, the glass spin model has shape (NxN))

    Inputs:
    - sg_model: type SG_model, specifies the spin glass model   
    - perturb_ones: array with dimensions (NxN), specifies a perturbation of each
        node's potential when the node takes the value 1
    - perturb_zeros:  array with dimensions (NxN), specifies a perturbation of each
        node's potential when the node takes the value 0 (really -1)
    - state: binary (0,1) array with dimensions (NxN), we are finding the perturbed
        energy of this particular state. state[i,j] = 0 means node ij takes the value -1,
        state[i,j] = 1 means node ij takes the value 1

    Outputs:
    - perturbed_energy: float, the perturbed energy of the specified state in the 
        specified spin glass model
    '''

    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])

    perturbed_energy = 0.0

    #Add single node potential and perturbation contributions
    for r in range(N):
        for c in range(N):
            if state[r,c] == 1:
                perturbed_energy += sg_model.lcl_fld_params[r,c]
                perturbed_energy += perturb_ones[r,c]
            else:
                assert(state[r,c] == 0)
                perturbed_energy -= sg_model.lcl_fld_params[r,c]        
                perturbed_energy += perturb_zeros[r,c]

    #Add two node potential contributions
    edge_count = 0
    for r in range(N):
        for c in range(N):
            #from horizontal edge
            if c < N-1:
                perturbed_energy += sg_model.cpl_params_h[r,c]*(2*state[r,c]-1)*(2*state[r,c+1]-1)
                edge_count += 1
            #from vertical edge
            if r < N-1:
                perturbed_energy += sg_model.cpl_params_v[r,c]*(2*state[r,c]-1)*(2*state[r+1,c]-1)
                edge_count += 1
    assert(edge_count == 2*N*(N-1))

    return perturbed_energy


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
        perturb_ones = np.random.gumbel(loc=0.0, scale=1.0, size=(N,N))
        perturb_zeros = np.random.gumbel(loc=0.0, scale=1.0, size=(N,N))
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

    upper_bound = upper_bound/num_trials
    return upper_bound

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
############    a_max = delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_min


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
        log_Z_upper_bound = np.log((1-beta)/beta)*a - n*np.log(1-beta) + log_w
        return log_Z_upper_bound, beta

############    (upper_bound_w_min, beta_w_min) = upper_bound_Z(delta_bar, total_num_random_vals, log_w_min, "min", k_2)
    (upper_bound_w_max, beta_w_max) = upper_bound_Z(delta_bar, total_num_random_vals, log_w_max, "max", k_2)
############    if np.isnan(upper_bound_w_min):
############        upper_bound_w_min = np.PINF
    if np.isnan(upper_bound_w_max):
        upper_bound_w_max = np.PINF


############    upper_bound_opt_beta = min([upper_bound_w_min, upper_bound_w_max, upper_bound1])
############    upper_bound_opt_beta = upper_bound1
    upper_bound_opt_beta = min([upper_bound_w_max, upper_bound1])


    if (delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_max)/total_num_random_vals >= .5 or \
       (delta_bar + np.sqrt(6*total_num_random_vals/k_2) - log_w_max)/total_num_random_vals <= 1/(1+np.e):
       upper_bound_opt_beta = upper_bound1
    else:
        assert(upper_bound_opt_beta == upper_bound_w_max)
    return upper_bound_opt_beta
############    return(upper_bound1, upper_bound_w_max, upper_bound_opt_beta)



def upper_bound_Z_barvinok(sg_model):
    '''
    Upper bound the partition function of the specified spin glass model using
    vector perturbations (uniform random in {-1,1}^n) inspired by Barvinok

    Inputs:
    - sg_model: type SG_model, specifies the spin glass model

    Outputs:
    - upper_bound: type float, upper bound on ln(partition function)
    '''

    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])

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
    delta = calc_perturbed_energy(sg_model, perturb_ones, perturb_zeros, map_state)

#############    ########### DEBUGGING ###########
#############    (delta_exp, map_state_debug) = find_MAP_spin_glass(sg_model, perturb_ones, perturb_zeros)
#############    check_delta = np.log(delta_exp)
#############    assert(np.abs(check_delta - delta) < .0001)
#############
#############    #assert(cur_perturbed_energy == check_delta), (cur_perturbed_energy, check_delta)
#############    delta = check_delta
#############    ########### END DEBUGGING ###########


    #option 1 for upper bound on log(permanent)
    total_num_random_vals = N**2
    upper_bound1 = delta + np.sqrt(6*total_num_random_vals) + total_num_random_vals*np.log(1+1/np.e)




    #check optimize beta
    unperturbed_map_state = spin_glass_perturbed_MAP_state(sg_model, np.zeros((N,N)), np.zeros((N,N)))
    log_w_max = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), unperturbed_map_state)
    a_min = delta + np.sqrt(6*total_num_random_vals) - log_w_max
############    a_max = delta + np.sqrt(6*total_num_random_vals) - log_w_min


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
        a = delta + np.sqrt(6*n) - log_w
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

############    (upper_bound_w_min, beta_w_min) = upper_bound_Z(delta, total_num_random_vals, log_w_min, "min")
    (upper_bound_w_max, beta_w_max) = upper_bound_Z(delta, total_num_random_vals, log_w_max, "max")
############    if np.isnan(upper_bound_w_min):
############        upper_bound_w_min = np.PINF
    if np.isnan(upper_bound_w_max):
        upper_bound_w_max = np.PINF


############    upper_bound_opt_beta = min([upper_bound_w_min, upper_bound_w_max, upper_bound1])
############    upper_bound_opt_beta = upper_bound1
    upper_bound_opt_beta = min([upper_bound_w_max, upper_bound1])


    if (delta + np.sqrt(6*total_num_random_vals) - log_w_max)/total_num_random_vals >= .5 or \
       (delta + np.sqrt(6*total_num_random_vals) - log_w_max)/total_num_random_vals <= 1/(1+np.e):
       upper_bound_opt_beta = upper_bound1
    else:
        assert(upper_bound_opt_beta == upper_bound_w_max)
    return upper_bound_opt_beta
############    return(upper_bound1, upper_bound_w_max, upper_bound_opt_beta)


def eval_partition_func_spin_glass(sg_model):
    '''
    
    Inputs:
        -sg_model: type SG_model, the spin glass model whose partition
        function we are evaluating

    Outputs:
        -partition_function: float, the value of the partition function
    '''
    G = MarkovModel()

    #create an NxN grid of nodes
    node_names = ['x%d%d' % (r,c) for r in range(sg_model.N) for c in range(sg_model.N)]
#    print node_names
    G.add_nodes_from(node_names)

    #add an edge between each node and its 4 neighbors, except when the
    #node is on the grid border and has fewer than 4 neighbors
    edges = []
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            if r < sg_model.N-1:
                edges.append(('x%d%d' % (r,c), 'x%d%d' % (r+1,c)))
            if c < sg_model.N-1:
                edges.append(('x%d%d' % (r,c), 'x%d%d' % (r,c+1)))
    assert(len(edges) == 2*sg_model.N*(sg_model.N-1))
#    print edges
#    print "number edges =", len(edges)
    G.add_edges_from(edges) 

    all_factors = []
    #set single variable potentials
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            node_name = 'x%d%d' % (r,c) 
            theta_i = sg_model.lcl_fld_params[r,c]
            factor_vals = np.array([np.exp(-theta_i), np.exp(theta_i)])
            all_factors.append(DiscreteFactor([node_name], cardinality=[2], values=factor_vals))

    #set two variable potentials
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            if r < sg_model.N-1: #vertical edge
                edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r+1,c))
                theta_ij = sg_model.cpl_params_v[r,c]
                factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
                                        [np.exp(-theta_ij), np.exp( theta_ij)]])
                all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))

            if c < sg_model.N-1: #horizontal edge
                edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r,c+1))
                theta_ij = sg_model.cpl_params_h[r,c]
                factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
                                        [np.exp(-theta_ij), np.exp( theta_ij)]])
                all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))


    G.add_factors(*all_factors)
 
    partition_function = get_partition_function_BP(G)
    return partition_function




def find_MAP_spin_glass(sg_model, perturb_ones, perturb_zeros):
    '''
    Find the MAP state of the specified perturbed spin glass model using a
    general MRF MAP solver 
    Inputs:
        -sg_model: type SG_model, the spin glass model whose partition
        function we are evaluating
        - perturb_ones: array with dimensions (NxN), specifies a perturbation of each
            node's potential when the node takes the value 1
        - perturb_zeros:  array with dimensions (NxN), specifies a perturbation of each
            node's potential when the node takes the value 0 (really -1)

    Outputs:
        -partition_function: float, the value of the partition function
    '''
    G = MarkovModel()

    #create an NxN grid of nodes
    node_names = ['x%d%d' % (r,c) for r in range(sg_model.N) for c in range(sg_model.N)]
#    print node_names
    G.add_nodes_from(node_names)

    #add an edge between each node and its 4 neighbors, except when the
    #node is on the grid border and has fewer than 4 neighbors
    edges = []
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            if r < sg_model.N-1:
                edges.append(('x%d%d' % (r,c), 'x%d%d' % (r+1,c)))
            if c < sg_model.N-1:
                edges.append(('x%d%d' % (r,c), 'x%d%d' % (r,c+1)))
    assert(len(edges) == 2*sg_model.N*(sg_model.N-1))
#    print edges
#    print "number edges =", len(edges)
    G.add_edges_from(edges) 

    all_factors = []
    #set single variable potentials
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            node_name = 'x%d%d' % (r,c) 
            theta_i = sg_model.lcl_fld_params[r,c]
            factor_vals = np.array([np.exp(-theta_i + perturb_zeros[r,c]), np.exp(theta_i + perturb_ones[r,c])])
            all_factors.append(DiscreteFactor([node_name], cardinality=[2], values=factor_vals))

    #set two variable potentials
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            if r < sg_model.N-1: #vertical edge
                edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r+1,c))
                theta_ij = sg_model.cpl_params_v[r,c]
                factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
                                        [np.exp(-theta_ij), np.exp( theta_ij)]])
                all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))

            if c < sg_model.N-1: #horizontal edge
                edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r,c+1))
                theta_ij = sg_model.cpl_params_h[r,c]
                factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
                                        [np.exp(-theta_ij), np.exp( theta_ij)]])
                all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))


    G.add_factors(*all_factors)
 
    map_val = find_MAP_val(G)
    map_state = find_MAP_state(G)
    return (map_val, map_state)

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


def compare_upper_bounds(N, f, num_gumbel_trials, k_2):
    coupling_strengths = [i/10.0 for i in range(1, 100)]
#    coupling_strengths = [i/100.0 for i in range(900, 1000)]
#    coupling_strengths = [1]
    gumbel_UBs = []
    gumbel_errors = []
    barvinok_UBs = []
    barvinok_errors = []
    barvinok_UB1s = []
    barvinok_UBwmaxs = []
    exact_log_Zs = [] #exact values of log(partition function)

    gumbel_barvinok_diff = []
    for (idx, c) in enumerate(coupling_strengths):
        print idx/len(coupling_strengths), 'fraction of the way through'
        sg_model = SG_model(N, f, c)
        gumbel_UB = upper_bound_Z_gumbel(sg_model, num_gumbel_trials)
        gumbel_UBs.append(gumbel_UB)
        barvinok_UB = upper_bound_Z_barvinok_k2(sg_model, k_2=k_2)
        barvinok_UBs.append(barvinok_UB)

        gumbel_barvinok_diff.append(gumbel_UB - barvinok_UB)
#        (barvinok_UB1, barvinok_UB_w_max, barvinok_UB_min) = upper_bound_Z_barvinok(sg_model)
#        barvinok_UB1s.append(barvinok_UB1)
#        barvinok_UBwmaxs.append(barvinok_UB_w_max)
#        barvinok_UBs.append(barvinok_UB_min)

        CALC_EXACT_Z = False
        if CALC_EXACT_Z:    
            partition_function = eval_partition_func_spin_glass(sg_model)
            exact_log_Zs.append(np.log(partition_function))
    
            gumbel_errors.append(gumbel_UB - np.log(partition_function))
            barvinok_errors.append(barvinok_UB - np.log(partition_function))


    fig = plt.figure()
    ax = plt.subplot(111)


#    ax.plot(coupling_strengths, barvinok_UBs, 'bx', label='our upper bound min')
#    ax.plot(coupling_strengths, barvinok_UB1s, 'yx', label='our upper bound1')
#    ax.plot(coupling_strengths, barvinok_UBwmaxs, 'gx', label='our upper bound wmax')

    #plot our estimate, gumbel estimate, and exact log(Z)
#    ax.plot(coupling_strengths, barvinok_UBs, 'bx', label='our upper bound')
#    ax.plot(coupling_strengths, gumbel_UBs, 'rx', label='gumbel upper bound')
#    ax.plot(coupling_strengths, exact_log_Zs, 'g*', label='exact log(Z)')

    #plot our estimation error and gumbel estimation error
#    ax.plot(coupling_strengths, barvinok_errors, 'bx', label='our error')
#    ax.plot(coupling_strengths, gumbel_errors, 'rx', label='gumbel error')

   #plot difference (gumbel estimate - our estimate)
    ax.plot(coupling_strengths, gumbel_barvinok_diff, 'bx', label='gumbel_UB - our_UB')


    plt.title('ln(Z) upper bounds')
    plt.xlabel('coupling_strengths')
    plt.ylabel('ln(Z)')
    #plt.legend()
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('CUR_TEST_upper_bounds', bbox_extra_artists=(lgd,), bbox_inches='tight')    
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
    compare_upper_bounds(20, 1.0, 100, k_2=10)
#    compare_upper_bounds(100, 1.0, 10)

#    sg_model = SG_model(N=5, f=1.0, c=3.0)
#    print "Z =", eval_partition_func_spin_glass(sg_model)