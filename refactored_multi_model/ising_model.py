from __future__ import division
import numpy as np
import maxflow
import matplotlib
matplotlib.use('Agg') #prevent error running remotely
import matplotlib.pyplot as plt
import time

from pgmpy.models import MarkovModel

from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor
from itertools import product

from boundZ import lower_bound_Z as gen_lower_bound_Z
from boundZ import upper_bound_Z as gen_upper_bound_Z
from boundZ import calculate_gumbel_slack

#from boundZ import upper_bound_Z_corrected_log_base as gen_upper_bound_Z_corrected
class SG_model:
    def __init__(self, N, f, c, all_weights_1=False):
        '''
        Sample local field parameters and coupling parameters to define a spin glass model
        
        Inputs:
        - N: int, the model will be a grid with shape (NxN)
        - f: float, local field parameters (theta_i) will be drawn uniformly at random
            from [-f, f] for each node in the grid
        - c: float, coupling parameters (theta_ij) will be drawn uniformly at random from
            [0, c) (gumbel paper uses [0,c], but this shouldn't matter) for each edge in 
            the grid
        - all_weights_1: bool, if true return a model with all weights = 1 so Z=2^(N^2)
                               if false return randomly sampled model

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

        if all_weights_1: #make all weights 1
            #sample local field parameters (theta_i) for each node
            self.lcl_fld_params = np.zeros((N,N))
    
            #sample horizontal coupling parameters (theta_ij) for each horizontal edge
            self.cpl_params_h = np.zeros((N,N-1))
    
            #sample vertical coupling parameters (theta_ij) for each vertical edge
            self.cpl_params_v = np.zeros((N-1,N))

        else: #randomly sample weights
            #sample local field parameters (theta_i) for each node
            self.lcl_fld_params = np.random.uniform(low=-f, high=f, size=(N,N))
    
            #sample horizontal coupling parameters (theta_ij) for each horizontal edge
            self.cpl_params_h = np.random.uniform(low=0.0, high=c, size=(N,N-1))
    
            #sample vertical coupling parameters (theta_ij) for each vertical edge
            self.cpl_params_v = np.random.uniform(low=0.0, high=c, size=(N-1,N))


def spin_glass_perturbed_MAP_state(sg_model, perturb_ones, perturb_zeros, log_base=np.e):
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
    - log_base: we're finding max_x {log w(x) + f(x)} where f(x) is some perturbation
        to each state (e.g. <c,x> where c is a random vector or gamma(x) gumbel 
        perturbation).  This specifies the base of the log in the log w(x) term.  
        Should be e for gumbel perturbations and 2 for c in {-1,1}^n perturbations.

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

            lamda_i = (sg_model.lcl_fld_params[r,c]/np.log(log_base) + perturb_ones[r,c]) \
                    - (-sg_model.lcl_fld_params[r,c]/np.log(log_base) + perturb_zeros[r,c])


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
                g.add_edge(nodes[r*N + c], nodes[r*N + c+1], 2*sg_model.cpl_params_h[r,c]/np.log(log_base),
                           2*sg_model.cpl_params_h[r,c]/np.log(log_base))
                edge_count += 1
            #add a vertical edge
            if r < N-1:
                g.add_edge(nodes[r*N + c], nodes[(r+1)*N + c], 2*sg_model.cpl_params_v[r,c]/np.log(log_base),
                           2*sg_model.cpl_params_v[r,c]/np.log(log_base))
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

def calc_perturbed_energy(sg_model, perturb_ones, perturb_zeros, state, log_base=np.e):
    '''
    Compute the perturbed energy of the given state in a spin glass model.  We mean
    {log w(x) + f(x)} for the state x where f(x) is some perturbation, w(x)=Z*p(x), 
    and the log has has the specified base.
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
    - log_base: we're finding {log w(x) + f(x)} for some value of x, where f(x) is some 
        perturbation (e.g. <c,x> where c is a random vector or gamma(x) gumbel 
        perturbation).  This specifies the base of the log in the log w(x) term.  
        Should be e for gumbel perturbations and 2 for c in {-1,1}^n perturbations.

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
                perturbed_energy += sg_model.lcl_fld_params[r,c]/np.log(log_base)
                perturbed_energy += perturb_ones[r,c]
            else:
                assert(state[r,c] == 0)
                perturbed_energy -= sg_model.lcl_fld_params[r,c]/np.log(log_base)
                perturbed_energy += perturb_zeros[r,c]

    #Add two node potential contributions
    edge_count = 0
    for r in range(N):
        for c in range(N):
            #from horizontal edge
            if c < N-1:
                perturbed_energy += (sg_model.cpl_params_h[r,c]/np.log(log_base))*(2*state[r,c]-1)*(2*state[r,c+1]-1)
                edge_count += 1
            #from vertical edge
            if r < N-1:
                perturbed_energy += (sg_model.cpl_params_v[r,c]/np.log(log_base))*(2*state[r,c]-1)*(2*state[r+1,c]-1)
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
        perturb_ones = np.random.gumbel(loc=-np.euler_gamma, scale=1.0, size=(N,N))
        perturb_zeros = np.random.gumbel(loc=-np.euler_gamma, scale=1.0, size=(N,N))
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
        cur_log_Z_est = np.max(energies+cur_gumbel_perturbations) - np.euler_gamma
        log_Z_estimates.append(cur_log_Z_est)
    return (np.mean(log_Z_estimates), exact_log_Z)


def expanded_spin_glass_perturbed_MAP_state(sg_model, copy_factor, perturb_ones, perturb_zeros):
    '''
    Find the MAP state of an expanded perturbed spin glass model using a min-cut/max-flow
    solver for the Gumbel lower bound on Z.  This method is faster than a MAP solver for a general 
    MRF, but can only be used for special cases of MRF's.

    For info about what models can be used see:
    http://www.cs.cornell.edu/~rdz/Papers/KZ-PAMI04.pdf

    For info about transforming the MAP problem into a min-cut problem see:
    https://www.classes.cs.uchicago.edu/archive/2006/fall/35040-1/gps.pdf

    (N is an implicit parameter, the glass spin model has shape (NxN))


    Inputs:
    - sg_model: type SG_model, specifies the (original, unexpanded) spin glass model
    - copy_factor: int, specifies the number of copies of each variable in the
        expanded model.  We will take the original (N x N) spin glass model and expand it
        to a model with dimensions (copy_factor*N x copy_factor*N), creating copy_factor^2
        copies of each variable
    - perturb_ones: array with dimensions (copy_factor*N x copy_factor*N), specifies a perturbation of each
        node's potential when the node takes the value 1
    - perturb_zeros:  array with dimensions (copy_factor*N x copy_factor*N), specifies a perturbation of each
        node's potential when the node takes the value 0

    Outputs:
    - map_state: binary array with dimensions (copy_factor*N x copy_factor*N), the state in the 
        expanded spin glass model with the largest energy.  map_state[i,j] = 0 means node (i,j) 
        is zero in the map state and map_state[i,j] = 1 means node (i,j) is one in the map state
    '''
    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])
    g = maxflow.GraphFloat()
    #Create (NxN) grid of nodes nodes
    nodes = g.add_nodes((copy_factor*N)**2)

    t0 = time.time()

    #Add single node potentials
    for r in range(N):
        for c in range(N):
            for cpy_idx_r in range(copy_factor):
                for cpy_idx_c in range(copy_factor):
                    lamda_i = (sg_model.lcl_fld_params[r,c] + perturb_ones[cpy_idx_r*N+r, cpy_idx_c*N+c]) \
                            - (-sg_model.lcl_fld_params[r,c] + perturb_zeros[cpy_idx_r*N+r, cpy_idx_c*N+c])


                    perturbed_0_potential = np.max((0, lamda_i))
                    perturbed_1_potential = np.max((0, -lamda_i))

        #            g.add_tedge(nodes[r*N + c], perturbed_0_potential, perturbed_1_potential)
                    g.add_tedge(nodes[(cpy_idx_r*N+r)*copy_factor*N + cpy_idx_c*N+c], perturbed_1_potential, perturbed_0_potential)

    t1 = time.time()
    print "adding single node potentials took:", t1-t0


    #DOUBLE CHECK INDEXING HERE!!
    #Add two node potentials
    edge_count = 0
    for r in range(N):
        for c in range(N):
            for cpy_idx_from_r in range(copy_factor):
                for cpy_idx_from_c in range(copy_factor):
                    for cpy_idx_to_r in range(copy_factor):
                        for cpy_idx_to_c in range(copy_factor):
                            #add a horizontal edge
                            if c < N-1:
                                g.add_edge(nodes[(cpy_idx_from_r*N + r)*copy_factor*N + cpy_idx_from_c*N + c], 
                                           nodes[(cpy_idx_to_r*N + r)*copy_factor*N + cpy_idx_to_c*N + c+1], 2*sg_model.cpl_params_h[r,c], 2*sg_model.cpl_params_h[r,c])
                                edge_count += 1
                            #add a vertical edge
                            if r < N-1:
                                g.add_edge(nodes[(cpy_idx_from_r*N + r)*copy_factor*N + cpy_idx_from_c*N + c], 
                                           nodes[(cpy_idx_to_r*N + r+1)*copy_factor*N + cpy_idx_to_c*N + c], 2*sg_model.cpl_params_v[r,c], 2*sg_model.cpl_params_v[r,c])
                                edge_count += 1
    assert(edge_count == 2*N*(N-1)*copy_factor**4), (edge_count, 2*N*(N-1)*copy_factor**4)

    t2 = time.time()
    print "adding double node potentials took:", t2-t1


    #find the maximum flow
    flow = g.maxflow()

    t3 = time.time()
    print "computing min cut took:", t3-t2


    #read out node partitioning for max-flow
    map_state = np.zeros((copy_factor*N,copy_factor*N))
    for r in range(copy_factor*N):
        for c in range(copy_factor*N):
            map_state[r,c] = g.get_segment(nodes[r*copy_factor*N + c])
            assert(map_state[r,c] == 0 or map_state[r,c] == 1)

    return map_state



def calc_expanded_perturbed_energy(sg_model, copy_factor, perturb_ones, perturb_zeros, state):
    '''
    Compute the perturbed energy of the given state in an expanded spin glass model
    (N is an implicit parameter, the glass spin model has shape (NxN))

    Inputs:
    - sg_model: type SG_model, specifies the spin glass model   
    - copy_factor: int, specifies the number of copies of each variable in the
        expanded model.  The original (N x N) spin glass model has been expanded
        to a model with dimensions (copy_factor*N x copy_factor*N), with copy_factor^2
        copies of each variable    
    - perturb_ones: array with dimensions (copy_factor*N x copy_factor*N), specifies a perturbation of each
        node's potential when the node takes the value 1
    - perturb_zeros:  array with dimensions (copy_factor*N x copy_factor*N), specifies a perturbation of each
        node's potential when the node takes the value 0 (really -1)
    - state: binary (0,1) array with dimensions (copy_factor*N x copy_factor*N), we are finding the perturbed
        energy of this particular state. state[i,j] = 0 means node ij takes the value -1,
        state[i,j] = 1 means node ij takes the value 1

    Outputs:
    - perturbed_energy: float, the perturbed energy of the specified state in the 
        specified spin glass model
    '''

    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])

    single_node_energy_contribution = 0.0

    #Add single node potential and perturbation contributions
    for r in range(N):
        for c in range(N):
            for cpy_idx_r in range(copy_factor):
                for cpy_idx_c in range(copy_factor):
                    if state[r,c] == 1:
                        single_node_energy_contribution += sg_model.lcl_fld_params[r,c]
                        single_node_energy_contribution += perturb_ones[cpy_idx_r*N+r, cpy_idx_c*N+c]

                    else:
                        assert(state[r,c] == 0)
                        single_node_energy_contribution -= sg_model.lcl_fld_params[r,c]
                        single_node_energy_contribution += perturb_zeros[cpy_idx_r*N+r, cpy_idx_c*N+c]
    single_node_energy_contribution/=copy_factor**2
    #Add two node potential contributions
    edge_count = 0


    two_node_energy_contribution = 0.0
    for r in range(N):
        for c in range(N):
            for cpy_idx_from_r in range(copy_factor):
                for cpy_idx_from_c in range(copy_factor):
                    for cpy_idx_to_r in range(copy_factor):
                        for cpy_idx_to_c in range(copy_factor):
                            #from horizontal edge
                            if c < N-1:
#                                g.add_edge(nodes[(cpy_idx_from_r*N + r)*copy_factor*N + cpy_idx_from_c*N + c], 
#                                           nodes[(cpy_idx_to_r*N + r)*copy_factor*N + cpy_idx_to_c*N + c+1], 2*sg_model.cpl_params_h[r,c], 2*sg_model.cpl_params_h[r,c])
                               
                                two_node_energy_contribution += sg_model.cpl_params_h[r,c]\
                                                  *(2*state[cpy_idx_from_r*N + r, cpy_idx_from_c*N + c]-1)\
                                                  *(2*state[cpy_idx_to_r*N + r,cpy_idx_to_c*N + c +1]-1)

                                edge_count += 1
                            #from vertical edge
                            if r < N-1:
#                                g.add_edge(nodes[(cpy_idx_from_r*N + r)*copy_factor*N + cpy_idx_from_c*N + c], 
#                                           nodes[(cpy_idx_to_r*N + r+1)*copy_factor*N + cpy_idx_to_c*N + c], 2*sg_model.cpl_params_v[r,c], 2*sg_model.cpl_params_v[r,c])
                                two_node_energy_contribution += sg_model.cpl_params_v[r,c]\
                                                  *(2*state[cpy_idx_from_r*N + r, cpy_idx_from_c*N + c]-1)\
                                                  *(2*state[cpy_idx_to_r*N + r+1, cpy_idx_to_c*N + c]-1)
                                edge_count += 1

    two_node_energy_contribution/=copy_factor**4

    assert(edge_count == 2*N*(N-1)*copy_factor**4)

################    #divide by product_{i=1}^n M_i
################    #where M_i = copy_factor^2 and n = N^2 
################    print perturbed_energy
################    perturbed_energy /= (copy_factor**2)**(N**2)
################    print perturbed_energy

    perturbed_energy = single_node_energy_contribution + two_node_energy_contribution

    return perturbed_energy

def lower_bound_Z_gumbel(sg_model, copy_factor):
    '''
    lower bound the partition function of the specified spin glass model using
    an empircal estimate of the expected maximum energy perturbed with 
    Gumbel noise, using the copy approach

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


    perturb_ones = np.random.gumbel(loc=-np.euler_gamma, scale=1.0, size=(copy_factor*N, copy_factor*N))
    perturb_zeros = np.random.gumbel(loc=-np.euler_gamma, scale=1.0, size=(copy_factor*N, copy_factor*N))

    expanded_map_state = expanded_spin_glass_perturbed_MAP_state(sg_model, copy_factor, perturb_ones, perturb_zeros)
    lower_bound = calc_expanded_perturbed_energy(sg_model, copy_factor, perturb_ones, perturb_zeros, expanded_map_state)

    return lower_bound

def lower_bound_Z_gumbel2(sg_model, num_trials):
    '''
    lower bound the partition function of the specified spin glass model using
    an empircal estimate of the expected maximum energy perturbed with 
    Gumbel noise divided by 1/n

    Inputs:
    - sg_model: type SG_model, specifies the spin glass model
    - num_trials: int, estimate the expected perturbed maximum energy as
        the mean over num_trials perturbed maximum energies


    Outputs:
    - lower_bound: type float, lower bound on ln(partition function)
    '''

    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])

    lower_bound = 0.0
    for i in range(num_trials):
        perturb_ones = np.random.gumbel(loc=-np.euler_gamma, scale=1.0, size=(N,N))/(N**2)
        perturb_zeros = np.random.gumbel(loc=-np.euler_gamma, scale=1.0, size=(N,N))/(N**2)
        map_state = spin_glass_perturbed_MAP_state(sg_model, perturb_ones, perturb_zeros)
        cur_perturbed_energy = calc_perturbed_energy(sg_model, perturb_ones, perturb_zeros, map_state)
        lower_bound += cur_perturbed_energy

#    lower_bound = lower_bound/num_trials
    lower_bound = lower_bound/num_trials
    return lower_bound


def upper_bound_Z_barvinok_k2(sg_model, k_2, log_base, compute_min_weight):
    '''
    Upper bound the partition function of the specified spin glass model using
    vector perturbations (uniform random in {-1,1}^n) inspired by Barvinok

    Inputs:
    - sg_model: type SG_model, specifies the spin glass model
    - k_2: int, take the mean of k_2 solutions to independently perturbed optimization
        problems to tighten the slack term between max and expected max
    - log_base: float.  We will return bounds and an estimator for log(Z) using this base
    - compute_min_weight: bool, should we compute the minimum weight and use to potentially
        improve the bounds?  This cannot be done efficiently in general (I think), here 
        we use a general MAP solver instead of min cut alogirthm

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
        map_state = spin_glass_perturbed_MAP_state(sg_model, perturb_ones, perturb_zeros, log_base=2)
        cur_delta = calc_perturbed_energy(sg_model, perturb_ones, perturb_zeros, map_state, log_base=2)

##Debuging
#        (check_map_val, check_map_state) = find_MAP_spin_glass(sg_model, perturb_ones, perturb_zeros)
#        assert(np.abs(np.log(check_map_val) - cur_delta)) < .0001, (np.log(check_map_val), cur_delta)
##End debugging

        deltas.append(cur_delta)

    delta_bar = np.mean(deltas)

    total_num_random_vals = N**2

    unperturbed_map_state = spin_glass_perturbed_MAP_state(sg_model, np.zeros((N,N)), np.zeros((N,N)))
    log_w_max = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), unperturbed_map_state)
    w_max = np.exp(log_w_max)
    w_min = None
    if compute_min_weight:
        (w_min, w_min_state) = find_w_min_spin_glass(sg_model)

###############testing
######    print '*'*80
######    (OLD_high_prob_upper_bound_base_e, OLD_expectation_upper_bound_base_e) = gen_upper_bound_Z(delta_bar=delta_bar, \
######        n=total_num_random_vals, k=k_2, log_base=np.e, w_max=w_max, w_min=w_min, verbose=True)
######
######
######    (NEW_rescaled_high_prob_best_upper_bound, NEW_rescaled_expectation_best_upper_bound) = gen_upper_bound_Z_corrected(delta_bar=delta_bar, \
######        n=total_num_random_vals, k=k_2, log_base=np.e, w_max=w_max, w_min=w_min, verbose=True)
######
######
######
######    print "OLD_high_prob_upper_bound_base_e=", OLD_high_prob_upper_bound_base_e
######    print "OLD_expectation_upper_bound_base_e=", OLD_expectation_upper_bound_base_e
######
######    print "NEW_rescaled_high_prob_best_upper_bound =", NEW_rescaled_high_prob_best_upper_bound
######    print "NEW_rescaled_expectation_best_upper_bound =", NEW_rescaled_expectation_best_upper_bound
######
######
######    sleep(3)
###############testing done

    
    (high_prob_upper_bound, expectation_upper_bound) = gen_upper_bound_Z(delta_bar=delta_bar, \
        n=total_num_random_vals, k=k_2, log_base=log_base, w_max=w_max, w_min=w_min, verbose=True)

    (high_prob_lower_bound, expectation_lower_bound) = gen_lower_bound_Z(delta_bar=delta_bar, \
        n=total_num_random_vals, k=k_2, log_base=log_base, w_min=w_min, verbose=True)

    scaled_delta_bar = delta_bar*np.log(2)/np.log(log_base)

    scaled_upper_guess = None

    return (high_prob_lower_bound, expectation_lower_bound, high_prob_upper_bound, expectation_upper_bound, scaled_delta_bar, scaled_upper_guess, log_w_max)





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
#    assert(np.abs(partition_function - G.get_partition_function()) < .001), (partition_function, G.get_partition_function())

    return partition_function

def find_w_min_spin_glass(sg_model):
    '''
    Find the state with the smallest probability in the specified spin 
    glass model using a general MRF MAP solver 
    Inputs:
        -sg_model: type SG_model, the spin glass model whose smallest weight
            we are finding (of size NxN)

    Outputs:
        -w_min: float, the smallest weight (note: actual weight, NOT log(weight))
        -w_min_state: array (NxN), state with the smallest weight w_min
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
    #set single variable potentials to inverse of potentials in specified spin glass model
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            node_name = 'x%d%d' % (r,c) 
            theta_i = sg_model.lcl_fld_params[r,c]
            factor_vals = np.array([np.exp(theta_i), np.exp(-theta_i)])
            all_factors.append(DiscreteFactor([node_name], cardinality=[2], values=factor_vals))

    #set two variable potentials to inverse of potentials in specified spin glass model
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            if r < sg_model.N-1: #vertical edge
                edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r+1,c))
                theta_ij = sg_model.cpl_params_v[r,c]
                factor_vals = np.array([[np.exp(-theta_ij), np.exp( theta_ij)],
                                        [np.exp( theta_ij), np.exp(-theta_ij)]])
                all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))

            if c < sg_model.N-1: #horizontal edge
                edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r,c+1))
                theta_ij = sg_model.cpl_params_h[r,c]
                factor_vals = np.array([[np.exp(-theta_ij), np.exp( theta_ij)],
                                        [np.exp( theta_ij), np.exp(-theta_ij)]])
                all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))


    G.add_factors(*all_factors)
 
    #find w_min, or the largest weight in G with inverse potentials from the specified spin glass model
    w_min = 1.0/find_MAP_val(G)
    w_min_state_dict = find_MAP_state(G)
    w_min_state = np.zeros((sg_model.N,sg_model.N))
    for r in range(sg_model.N):
        for c in range(sg_model.N):
            w_min_state[r, c] = w_min_state_dict["x%d%d" % (r, c)]
    return (w_min, w_min_state)

def test_find_w_min_spin_glass():
    '''
    For debugging, check find_w_min_spin_glass is correctly finding w_min and the corresponding state
    on small models
    '''
    N = 4
    sg_model = SG_model(N=N, f=1.0, c=1.5)

    all_states = [[0, 1] for i in range(N**2)]
    energies = np.zeros(2**(N**2))
    #- state: binary (0,1) array with dimensions (NxN), we are finding the perturbed
    #energy of this particular state. state[i,j] = 0 means node ij takes the value -1,
    #state[i,j] = 1 means node ij takes the value 1
    idx = 0

    w_min_explicit = None #will be tuple containing (w_min, w_min_state) that we find by enumeration
    for cur_state in product(*all_states):
        cur_state_array = np.zeros((N,N))
        for r in range(N):
            for c in range(N):
                cur_state_array[r,c] = cur_state[r*N + c]
        cur_energy = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), cur_state_array)

        if w_min_explicit == None:
            w_min_explicit = (np.exp(cur_energy), cur_state_array)
        elif np.exp(cur_energy) < w_min_explicit[0]:
            w_min_explicit = (np.exp(cur_energy), cur_state_array)
        idx += 1

    assert(idx == 2**(N**2))

    (w_min, w_min_state) = find_w_min_spin_glass(sg_model)
    assert(np.abs(w_min - w_min_explicit[0]) < .0001), (w_min, w_min_explicit[0])
    print w_min_state
    print w_min_explicit[1]
    assert(np.all(w_min_state == w_min_explicit[1])), (w_min_state, w_min_explicit[1])
    print "find_w_min_spin_glass found the correct w_min!"



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
WRITE ME

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


def compare_upper_bounds(N, f, num_gumbel_trials, k_2, gumbel_LB_copy_factor):
    CALC_EXACT_Z = True
    CALC_GUMBEL_LB = True

    print "gumbel_LB_copy_factor =", gumbel_LB_copy_factor
    coupling_strengths = [i/10.0 for i in range(1, 30)]
#    coupling_strengths = [i/10.0 for i in range(1, 10)]
#    coupling_strengths = [i/10.0 for i in range(12, 13)]
#    coupling_strengths = [i/100.0 for i in range(900, 1000)]
#    coupling_strengths = [1]
    expectation_gumbel_UBs = []
    high_prob_gumbel_UBs = []
    gumbel_errors = []
    expectation_gumbel_LBs = []
    high_prob_gumbel_LBs = []

    barvinok_UBs = []
    barvinok_LBs = []

    expectation_barvinok_UBs = []
    expectation_barvinok_LBs = []

    guess_UBs = []

    barvinok_errors = []
    exact_log_Zs = [] #exact values of log(partition function)
    exact_log_Z2s = [] #debug, another list of exact values of log(partition function)

    our_estimators = [] #our estimators, delta_bar, of the partition function

    iid_gumbel_estimates = []

    gumbel_barvinok_diff = []

    log_w_maxs = []

    ##differences between upper and lower bounds on log(Z)
    our_high_prob_bound_width_log_Z = []
    our_expectation_bound_width_log_Z = []
    gumbel_high_prob_bound_width_log_Z = []
    gumbel_expectation_bound_width_log_Z = []

    for (idx, c) in enumerate(coupling_strengths):
        print idx/len(coupling_strengths), 'fraction of the way through'
        sg_model = SG_model(N, f, c, all_weights_1=False)
        gumbel_UB = upper_bound_Z_gumbel(sg_model, num_gumbel_trials)
        expectation_gumbel_UBs.append(gumbel_UB) 
        gumbel_slack = calculate_gumbel_slack(A=N**2, M=num_gumbel_trials, delta=.05)
        high_prob_gumbel_UB = gumbel_UB + gumbel_slack
        high_prob_gumbel_UBs.append(high_prob_gumbel_UB)
        if CALC_GUMBEL_LB:       
#            gumbel_LB = lower_bound_Z_gumbel(sg_model, gumbel_LB_copy_factor)
            gumbel_LB = lower_bound_Z_gumbel2(sg_model, num_gumbel_trials)
            expectation_gumbel_LBs.append(gumbel_LB)

            high_prob_gumbel_LB = gumbel_LB - gumbel_slack/(N**2)
            high_prob_gumbel_LBs.append(high_prob_gumbel_LB)

            gumbel_high_prob_bound_width_log_Z.append(high_prob_gumbel_UB - high_prob_gumbel_LB)
            gumbel_expectation_bound_width_log_Z.append(gumbel_UB - gumbel_LB)

        (high_prob_barvinok_LB, expectation_barvinok_LB, high_prob_barvinok_UB, expectation_barvinok_UB, \
            delta_bar, upper_guess, log_w_max) = upper_bound_Z_barvinok_k2(sg_model, k_2=k_2, log_base=np.e, compute_min_weight=False)
        
        our_estimators.append(delta_bar)

        barvinok_UBs.append(high_prob_barvinok_UB)
        barvinok_LBs.append(max(high_prob_barvinok_LB, log_w_max))
        expectation_barvinok_UBs.append(expectation_barvinok_UB)
        expectation_barvinok_LBs.append(expectation_barvinok_LB)
        our_high_prob_bound_width_log_Z.append(high_prob_barvinok_UB - high_prob_barvinok_LB)
        our_expectation_bound_width_log_Z.append(expectation_barvinok_UB - expectation_barvinok_LB)

        log_w_maxs.append(log_w_max)

        guess_UBs.append(upper_guess)

        gumbel_barvinok_diff.append(gumbel_UB - high_prob_barvinok_UB)

#        (iid_gumbel_estimate, exact_log_Z2) = estimate_Z_iid_gumbel(sg_model, 1000)
#        iid_gumbel_estimates.append(iid_gumbel_estimate)
#        exact_log_Z2s.append(exact_log_Z2)


        if CALC_EXACT_Z:    
            partition_function = eval_partition_func_spin_glass(sg_model)
            exact_log_Zs.append(np.log(partition_function))
    
            gumbel_errors.append(gumbel_UB - np.log(partition_function))
            barvinok_errors.append(high_prob_barvinok_UB - np.log(partition_function))

            print "Z =", partition_function
            print "w_max =", np.exp(log_w_max)
            print "Z/w_max =", partition_function/np.exp(log_w_max)

##########plot HIGH PROBABILITY estimates of log(Z)##########
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(coupling_strengths, barvinok_UBs, ':b+', label=r'$\psi_{UB}$', markersize=15)

    ax.plot(coupling_strengths, our_estimators, 'bx', label=r'$\bar{\delta}_k(w)$', markersize=15)

#    ax.plot(coupling_strengths, guess_UBs, 'm+', label='guess upper', markersize=15)


    ax.plot(coupling_strengths, high_prob_gumbel_UBs, '--r+', label=r'$\theta_{UB}$', markersize=15)
    if CALC_GUMBEL_LB:
        ax.plot(coupling_strengths, high_prob_gumbel_LBs, '--r2', label=r'$\theta_{LB}$', markersize=15)
    if CALC_EXACT_Z:
        ax.plot(coupling_strengths, exact_log_Zs, 'g*', label=r'$\ln(Z)$', markersize=10)

#    ax.plot(coupling_strengths, iid_gumbel_estimates, 'r+', label='iid gumbel estimate')
#    ax.plot(coupling_strengths, exact_log_Z2s, 'gx', label='exact log(Z2)')

    #plot our estimation error and gumbel estimation error
#    ax.plot(coupling_strengths, barvinok_errors, 'bx', label='our error')
#    ax.plot(coupling_strengths, gumbel_errors, 'rx', label='gumbel error')

   #plot difference (gumbel estimate - our estimate)
#    ax.plot(coupling_strengths, gumbel_barvinok_diff, 'bx', label='gumbel_UB - our_UB')
    ax.plot(coupling_strengths, barvinok_LBs, ':b2', label=r'$\psi_{LB}$', markersize=15)

    
    plt.title('High Probability Bounds on Z(w)')
    plt.xlabel('c (Maximum Coupling Parameter)')
    plt.ylabel('ln(Z)')
    #plt.legend()

    #lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #make the font bigger
    matplotlib.rcParams.update({'font.size': 15})

#    ax.set_xticks(np.arange(0, 1, 0.1))
#    ax.set_yticks(np.arange(0, 1., 0.1))
    plt.grid()

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
#    # Put a legend below current axis
#    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
#              fancybox=False, shadow=False, ncol=2, numpoints = 1)

    # Put a legend below current axis
    lgd = ax.legend(loc='upper left', fancybox=False, shadow=False, ncol=2, numpoints = 1, labelspacing=0.0001)


    plot_name = 'ising_model_highProb_wmaxLB_%dx%dmodel,gumbel_trials=%d,our_k=%d' % (N, N, num_gumbel_trials, k_2)

    fig.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()    # close the figure


##########plot EXPECTATION estimates of log(Z)##########
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(coupling_strengths, expectation_barvinok_UBs, 'g+', label='our UB, expectation', markersize=10)
    ax.plot(coupling_strengths, expectation_barvinok_LBs, 'g^', label='our LB, expectation', markersize=7)

    ax.plot(coupling_strengths, our_estimators, 'bx', label='our estimator', markersize=10)

#    ax.plot(coupling_strengths, guess_UBs, 'm+', label='guess upper', markersize=10)


    ax.plot(coupling_strengths, expectation_gumbel_UBs, 'r+', label='gumbel UB, expectation', markersize=10)
    ax.plot(coupling_strengths, log_w_maxs, 'mx', label='log w_max', markersize=10)
    if CALC_GUMBEL_LB:
        ax.plot(coupling_strengths, expectation_gumbel_LBs, 'r^', label='gumbel LB, expectation', markersize=7)
    if CALC_EXACT_Z:
        ax.plot(coupling_strengths, exact_log_Zs, 'g*', label='exact log(Z)', markersize=5)

#    ax.plot(coupling_strengths, iid_gumbel_estimates, 'r+', label='iid gumbel estimate')
#    ax.plot(coupling_strengths, exact_log_Z2s, 'gx', label='exact log(Z2)')

    #plot our estimation error and gumbel estimation error
#    ax.plot(coupling_strengths, barvinok_errors, 'bx', label='our error')
#    ax.plot(coupling_strengths, gumbel_errors, 'rx', label='gumbel error')

   #plot difference (gumbel estimate - our estimate)
#    ax.plot(coupling_strengths, gumbel_barvinok_diff, 'bx', label='gumbel_UB - our_UB')

    
    plt.title('Expectation Bounds on Z(w)')
    plt.xlabel('c (Maximum Coupling Parameter)')
    plt.ylabel('ln(Z)')
    #plt.legend()

    #lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #make the font bigger
    matplotlib.rcParams.update({'font.size': 15})

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)


    plot_name = 'ising_model_expectation_%dx%dmodel,gumbel_trials=%d,our_k=%d' % (N, N, num_gumbel_trials, k_2)
    fig.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()    # close the figure



############Plot gap differences between upper and lower bounds on log(Z)############

    fig = plt.figure()
    ax = plt.subplot(111)


    ax.plot(coupling_strengths, our_high_prob_bound_width_log_Z, 'b+', label='our high probability width', markersize=10)
    ax.plot(coupling_strengths, our_expectation_bound_width_log_Z, 'g+', label='our expectation width', markersize=7)
    ax.plot(coupling_strengths, gumbel_high_prob_bound_width_log_Z, 'm+', label='gumbel high probability width', markersize=10)
    ax.plot(coupling_strengths, gumbel_expectation_bound_width_log_Z, 'r+', label='gumbel expectation width', markersize=7)


    
    plt.title('Bound width on log(Z), %dx%d model, f=%f, gumbel_trials=%d, our k=%d' % (N, N, f, num_gumbel_trials, k_2))
    plt.xlabel('c (Maximum Coupling Parameter)')
    plt.ylabel('UB - LB')
    #plt.legend()

    #lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #make the font bigger
    matplotlib.rcParams.update({'font.size': 15})

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)

    fig.savefig('ising_model_bound_width', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()    # close the figure


##########    fig = plt.figure()
##########    ax = plt.subplot(111)
##########
##########    #plot our estimate, gumbel estimate, and exact log(Z)
##########    ax.plot(coupling_strengths, barvinok_UBs, 'bx', label='our upper bound')
##########    ax.plot(coupling_strengths, expectation_gumbel_UBs, 'rx', label='gumbel upper bound')
##########    ax.plot(coupling_strengths, exact_log_Zs, 'g*', label='exact log(Z)')
##########
##########
##########
##########    plt.title('ln(Z) upper bounds')
##########    plt.xlabel('c (Maximum Coupling Parameter)')
##########    plt.ylabel('ln(Z)')
##########    #plt.legend()
##########    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
##########    fig.savefig('CUR_TEST_upper_bounds1', bbox_extra_artists=(lgd,), bbox_inches='tight')    
##########    plt.close()    # close the figure


if __name__=="__main__":
#    for i in range(50):
#        test_find_w_min_spin_glass()


    compare_upper_bounds(N=7, f=1.0, num_gumbel_trials=5, k_2=5, gumbel_LB_copy_factor=10)

#    compare_upper_bounds(N=100, f=1.0, num_gumbel_trials=10, k_2=10, gumbel_LB_copy_factor=70)

#    compare_upper_bounds(100, 1.0, 10)

#    sg_model = SG_model(N=5, f=1.0, c=3.0)
#    print "Z =", eval_partition_func_spin_glass(sg_model)