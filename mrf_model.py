from __future__ import division
import numpy as np
import maxflow
import matplotlib.pyplot as plt
import time

from pgmpy.models import MarkovModel

from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor
from itertools import product


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
    def __init__(self, N, f, c, neighbor_count):
        '''
        Sample local field parameters and coupling parameters to define a spin glass model
        
        Inputs:
        - N: int, the model will be a grid with shape (NxN)
        - f: float, local field parameters (theta_i) will be drawn uniformly at random
            from [-f, f] for each node in the grid
        - c: float, coupling parameters (theta_ij) will be drawn uniformly at random from
            [0, c) (gumbel paper uses [0,c], but this shouldn't matter) for each edge in 
            the grid
        - grid_jumps: int, we'll create an MRF in a grid structure where each node has
            horizontal and vertical connections to its neighbor_count nearest neighbor
            in all 4 directions, except when too close to an edge

        Values defining the spin glass model:
        - lcl_fld_params: array with dimensions (NxN), local field parameters (theta_i)
            that we sampled for each node in the grid 
        - cpl_params_h: list of arrays with dimensions (N x N-1), (N x N-2),...,(N x N - neighbor_count).
            coupling parameters (theta_ij), for each horizontal edge in the grid. 
            cpl_params_h[m][k,l] corresponds to 
            theta_ij where i is the node indexed by (k,l) and j is the node indexed by
            (k,l+1+m)

        - cpl_params_v: list of arrays with dimensions (N-1 x N), (N-2 x N),...,(N - neighbor_count x N).
            coupling parameters (theta_ij), for each horizontal edge in the grid. 
            cpl_params_v[m][k,l] corresponds to 
            theta_ij where i is the node indexed by (k,l) and j is the node indexed by
            (k+1+m,l)
    
        '''
        self.N = N

        #sample local field parameters (theta_i) for each node
        self.lcl_fld_params = np.random.uniform(low=-f, high=f, size=(N,N))

        self.cpl_params_h = []
        self.cpl_params_v = []

        for i in range(neighbor_count):
            #sample horizontal coupling parameters (theta_ij) for each horizontal edge
            self.cpl_params_h.append(np.random.uniform(low=0.0, high=c, size=(N,N-1-i)))

            #sample vertical coupling parameters (theta_ij) for each vertical edge
            self.cpl_params_v.append(np.random.uniform(low=0.0, high=c, size=(N-1-i,N)))


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
    assert(len(sg_model.cpl_params_h) == len(sg_model.cpl_params_v))
    edge_count = 0
    for n_idx in range(len(sg_model.cpl_params_h)):
        cur_cpl_params_h = sg_model.cpl_params_h[n_idx]    
        cur_cpl_params_v = sg_model.cpl_params_v[n_idx]

        for r in range(N):
            for c in range(N):
                #add a horizontal edge
                if c < N-1-n_idx:
                    g.add_edge(nodes[r*N + c], nodes[r*N + c+1+n_idx], 2*cur_cpl_params_h[r,c], 2*cur_cpl_params_h[r,c])
                    edge_count += 1
                #add a vertical edge
                if r < N-1-n_idx:
                    g.add_edge(nodes[r*N + c], nodes[(r+1+n_idx)*N + c], 2*cur_cpl_params_v[r,c], 2*cur_cpl_params_v[r,c])
                    edge_count += 1
    check_edge_count = 0
    for n_idx in range(len(sg_model.cpl_params_h)):
        check_edge_count +=  2*N*(N-1 - n_idx)

    assert(edge_count == check_edge_count)

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
    assert(len(sg_model.cpl_params_h) == len(sg_model.cpl_params_v))
    edge_count = 0
    for n_idx in range(len(sg_model.cpl_params_h)):
        cur_cpl_params_h = sg_model.cpl_params_h[n_idx]    
        cur_cpl_params_v = sg_model.cpl_params_v[n_idx]

        for r in range(N):
            for c in range(N):
                #add a horizontal edge
                if c < N-1-n_idx:
                    perturbed_energy += cur_cpl_params_h[r,c]*(2*state[r,c]-1)*(2*state[r,c+1]-1)
                    edge_count += 1
                #add a vertical edge
                if r < N-1-n_idx:
                    perturbed_energy += cur_cpl_params_v[r,c]*(2*state[r,c]-1)*(2*state[r+1,c]-1)
                    edge_count += 1
    check_edge_count = 0
    for n_idx in range(len(sg_model.cpl_params_h)):
        check_edge_count +=  2*N*(N-1 - n_idx)


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

#HASN"T BEEN ADJUSTED FROM SPIN GLASS MODEL YET!!
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


#HASN"T BEEN ADJUSTED FROM SPIN GLASS MODEL YET!!
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



def enumerate_wishful_term_rademacher_LB(sg_model, perturb_ones, perturb_zeros, lamda):
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
        cur_energy = calc_perturbed_energy(sg_model, perturb_ones, perturb_zeros, cur_state_array)

        energies[idx] = cur_energy
        
#        print '-'*30
#        print "idx =", idx
#        print "cur_energy", cur_energy
#        print "energies", energies

        idx += 1

    assert(idx == 2**(N**2))

    S_1 = 0.0
    for cur_energy in energies:
#        print "cur_energy", cur_energy
        S_1 += np.exp(lamda*cur_energy)

    return S_1


def bound_Z_barvinok_wishful(sg_model, k_2):
    '''
    Test unproven bounds to get an optimistic idea of how bounds
    might compare with gumbel
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

    total_num_random_vals = N**2
##    #option 1 for upper bound on log(permanent)
##    total_num_random_vals = N**2
##    upper_bound1 = delta_bar + np.sqrt(6*total_num_random_vals/k_2) + total_num_random_vals*np.log(1+1/np.e)
##    
    wishful_LB = ((delta_bar - np.sqrt(6*total_num_random_vals/k_2))**2)/(2*total_num_random_vals)



    #corrected rademacher lower bound
    #check optimize beta
    unperturbed_map_state = spin_glass_perturbed_MAP_state(sg_model, np.zeros((N,N)), np.zeros((N,N)))
    log_w_max = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), unperturbed_map_state)
    (w_min, w_min_state) = find_w_min_spin_glass(sg_model)
    log_w_min = np.log(w_min)
    
    lambda_prime_w_max = (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_max)/total_num_random_vals
    lambda_prime_w_min = (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_min)/total_num_random_vals
    print "lambda_prime_w_max =", lambda_prime_w_max
    print "lambda_prime_w_min =", lambda_prime_w_min
    #check whether can we use w_max?
    if lambda_prime_w_max >= 1:
        print ":) :) we can use w_max for corrected rademacher!!"
        print "lambda =", lambda_prime_w_max
        wishful_corrected_rademacher_LB = (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_max)**2/(2*total_num_random_vals) + log_w_max


        s_1 = enumerate_wishful_term_rademacher_LB(sg_model, perturb_ones, perturb_zeros, lambda_prime_w_max)
        perturbed_MAP_state = spin_glass_perturbed_MAP_state(sg_model, np.zeros((N,N)), np.zeros((N,N)))
        perturbed_log_w_max = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), perturbed_MAP_state)        
        s_1_minus_s_2 = np.exp(lambda_prime_w_max*perturbed_log_w_max) 
        wishful_corrected_rademacher_LB -= lambda_prime_w_max*np.log(s_1_minus_s_2/s_1)

    #check whether can we use w_min?
    elif lambda_prime_w_min <= 1:
        print ":) we can use w_min for corrected rademacher!!"
        print "lambda =", lambda_prime_w_min
        print "delta_bar - np.sqrt(6*total_num_random_vals/k_2) =", delta_bar - np.sqrt(6*total_num_random_vals/k_2)
        print "log_w_min =", log_w_min
        print "2*total_num_random_vals =", 2*total_num_random_vals
        print "(delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_min)**2/(2*total_num_random_vals) =", (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_min)**2/(2*total_num_random_vals)
        wishful_corrected_rademacher_LB = (delta_bar - np.sqrt(6*total_num_random_vals/k_2) - log_w_min)**2/(2*total_num_random_vals) + log_w_min

        s_1 = enumerate_wishful_term_rademacher_LB(sg_model, perturb_ones, perturb_zeros, lambda_prime_w_min)
        perturbed_MAP_state = spin_glass_perturbed_MAP_state(sg_model, np.zeros((N,N)), np.zeros((N,N)))
        perturbed_log_w_max = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), perturbed_MAP_state)        
        s_1_minus_s_2 = np.exp(lambda_prime_w_min*perturbed_log_w_max) 
        wishful_corrected_rademacher_LB -= lambda_prime_w_min*np.log(s_1_minus_s_2/s_1)

    else:
        print ":( we can't use w_min or w_max for corrected rademacher"
        wishful_corrected_rademacher_LB = delta_bar - np.sqrt(6*total_num_random_vals/k_2) - total_num_random_vals/2

        s_1 = enumerate_wishful_term_rademacher_LB(sg_model, perturb_ones, perturb_zeros, 1.0)
        perturbed_MAP_state = spin_glass_perturbed_MAP_state(sg_model, np.zeros((N,N)), np.zeros((N,N)))
        perturbed_log_w_max = calc_perturbed_energy(sg_model, np.zeros((N,N)), np.zeros((N,N)), perturbed_MAP_state)        
        s_1_minus_s_2 = np.exp(1.0*perturbed_log_w_max) 
        wishful_corrected_rademacher_LB -= 1.0*np.log(s_1_minus_s_2/s_1)


    return (wishful_corrected_rademacher_LB, 0.0)
#    return (wishful_LB, 0.0)

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

    assert(len(sg_model.cpl_params_h) == len(sg_model.cpl_params_v))
    for n_idx in range(len(sg_model.cpl_params_h)):
        for r in range(sg_model.N):
            for c in range(sg_model.N):
                if r < sg_model.N-1-n_idx:
                    edges.append(('x%d%d' % (r,c), 'x%d%d' % (r+1 + n_idx,c)))
                if c < sg_model.N-1-n_idx:
                    edges.append(('x%d%d' % (r,c), 'x%d%d' % (r,c+1 + n_idx)))

    check_edge_count = 0
    for n_idx in range(len(sg_model.cpl_params_h)):
        check_edge_count +=  2*sg_model.N*(sg_model.N-1 - n_idx)

    assert(len(edges) == check_edge_count)

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
    for n_idx in range(len(sg_model.cpl_params_h)):
        for r in range(sg_model.N):
            for c in range(sg_model.N):
                if r < sg_model.N-1 - n_idx: #vertical edge
                    edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r+1 + n_idx,c))
                    theta_ij = sg_model.cpl_params_v[n_idx][r,c]
                    factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
                                            [np.exp(-theta_ij), np.exp( theta_ij)]])
                    all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))

                if c < sg_model.N-1 - n_idx: #horizontal edge
                    edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r,c+1 + n_idx))
                    theta_ij = sg_model.cpl_params_h[n_idx][r,c]
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

    assert(len(sg_model.cpl_params_h) == len(sg_model.cpl_params_v))
    for n_idx in range(len(sg_model.cpl_params_h)):
        for r in range(sg_model.N):
            for c in range(sg_model.N):
                if r < sg_model.N-1-n_idx:
                    edges.append(('x%d%d' % (r,c), 'x%d%d' % (r+1 + n_idx,c)))
                if c < sg_model.N-1-n_idx:
                    edges.append(('x%d%d' % (r,c), 'x%d%d' % (r,c+1 + n_idx)))

    check_edge_count = 0
    for n_idx in range(len(sg_model.cpl_params_h)):
        check_edge_count +=  2*sg_model.N*(sg_model.N-1 - n_idx)

    assert(len(edges) == check_edge_count)
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


    #set two variable potentials
    for n_idx in range(len(sg_model.cpl_params_h)):
        for r in range(sg_model.N):
            for c in range(sg_model.N):
                if r < sg_model.N-1 - n_idx: #vertical edge
                    edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r+1 + n_idx,c))
                    theta_ij = sg_model.cpl_params_v[n_idx][r,c]
                    factor_vals = np.array([[np.exp(-theta_ij), np.exp( theta_ij)],
                                            [np.exp( theta_ij), np.exp(-theta_ij)]])
                    all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))

                if c < sg_model.N-1 - n_idx: #horizontal edge
                    edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r,c+1 + n_idx))
                    theta_ij = sg_model.cpl_params_h[n_idx][r,c]
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
    sg_model = SG_model(N=N, f=1.0, c=1.5, neighbor_count=3)

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

    assert(len(sg_model.cpl_params_h) == len(sg_model.cpl_params_v))
    for n_idx in range(len(sg_model.cpl_params_h)):
        for r in range(sg_model.N):
            for c in range(sg_model.N):
                if r < sg_model.N-1-n_idx:
                    edges.append(('x%d%d' % (r,c), 'x%d%d' % (r+1 + n_idx,c)))
                if c < sg_model.N-1-n_idx:
                    edges.append(('x%d%d' % (r,c), 'x%d%d' % (r,c+1 + n_idx)))

    check_edge_count = 0
    for n_idx in range(len(sg_model.cpl_params_h)):
        check_edge_count +=  2*sg_model.N*(sg_model.N-1 - n_idx)

    assert(len(edges) == check_edge_count)
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
    for n_idx in range(len(sg_model.cpl_params_h)):
        for r in range(sg_model.N):
            for c in range(sg_model.N):
                if r < sg_model.N-1 - n_idx: #vertical edge
                    edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r+1 + n_idx,c))
                    theta_ij = sg_model.cpl_params_v[n_idx][r,c]
                    factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
                                            [np.exp(-theta_ij), np.exp( theta_ij)]])
                    all_factors.append(DiscreteFactor(edge_name, cardinality=[2,2], values=factor_vals))

                if c < sg_model.N-1 - n_idx: #horizontal edge
                    edge_name = ('x%d%d' % (r,c), 'x%d%d' % (r,c+1 + n_idx))
                    theta_ij = sg_model.cpl_params_h[n_idx][r,c]
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