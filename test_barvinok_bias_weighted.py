from __future__ import division
from itertools import product
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevent error running remotely
import matplotlib.pyplot as plt
import math

import sys
sys.path.append("./refactored_multi_model")
from boundZ import lower_bound_Z as gen_lower_bound_Z
from boundZ import upper_bound_Z as gen_upper_bound_Z
from boundZ import upper_bound_Z_conjecture as gen_upper_bound_Z_conjecture


def generate_set(m, n, random):
    '''
    generate a set of m vectors in {-1,1}^n

    Inputs:
    - m: int, number of vectors
    - n: int, dimension of state space
    - random: bool, if true generate random vectors,
        if false generate hypercube requiring that m is a power of two and then
        generating all 2^m length m vectors with (n-m) 1's appended
        to each

    Outputs:
    - vector_set: list of tuples of ints (each -1 or 1).  Outer list
        is the set of vectors, each inner tuple is a particular
        vector in the set.

    '''
    if random:
        all_possible_vectors = [] #we're going to generate every vector in {-1,1}^n
        all_vectors_helper = [[-1, 1] for i in range(n)]
        idx = 0
        for cur_vector in product(*all_vectors_helper):
            all_possible_vectors.append(cur_vector)
        assert(len(all_possible_vectors) == 2**n)
        set_indices = np.random.choice(len(all_possible_vectors), size=m, replace=False)
        vector_set = [all_possible_vectors[i] for i in set_indices]
    else:
        assert(math.log(m)/math.log(2) - round(math.log(m)/math.log(2)) == 0) #make sure m is a power of 2
        log_2_m = int(math.log(m)/math.log(2))
        assert(n >= log_2_m)
        vectors_helper = [[-1, 1] for i in range(log_2_m)]
        vector_set = []
        for cur_partial_vector in product(*vectors_helper):
            cur_vector = cur_partial_vector + tuple([1 for i in range(n - log_2_m)])
            assert(len(cur_vector) == n)
            vector_set.append(cur_vector)

    assert(len(vector_set) == m)
    if len(vector_set) > len(set(vector_set)):
        assert(False), "Error, have an element in the vector set that isn't unique"

    return vector_set


def generate_set_efficient(m, n, random):
    '''
    generate a set of m vectors in {-1,1}^n without enumerating all 2^n vectors.
    Instead sample each vector uniformly at random from all 2^n possibilities and
    then check if we've sampled it already.

    Inputs:
    - m: int, number of vectors
    - n: int, dimension of state space
    - random: bool, if true generate random vectors,
        if false generate hypercube requiring that m is a power of two and then
        generating all 2^m length m vectors with (n-m) 1's appended
        to each

    Outputs:
    - vector_set: list of tuples of ints (each -1 or 1).  Outer list
        is the set of vectors, each inner tuple is a particular
        vector in the set.

    '''
    if random:
        vector_set = []
        while(len(vector_set) < m):
            cur_vector = tuple([-1 if (np.random.rand()<.5) else 1 for i in range(n)])
            if not cur_vector in vector_set:
                vector_set.append(cur_vector)
    else:
        assert(math.log(m)/math.log(2) - round(math.log(m)/math.log(2)) == 0) #make sure m is a power of 2
        log_2_m = int(math.log(m)/math.log(2))
        assert(n >= log_2_m)
        vectors_helper = [[-1, 1] for i in range(log_2_m)]
        vector_set = []
        for cur_partial_vector in product(*vectors_helper):
            cur_vector = cur_partial_vector + tuple([1 for i in range(n - log_2_m)])
            assert(len(cur_vector) == n)
            vector_set.append(cur_vector)

    assert(len(vector_set) == m)
    if len(vector_set) > len(set(vector_set)):
        assert(False), "Error, have an element in the vector set that isn't unique"

    return vector_set



def summary_stats_random_c(n, vector_set, randomness='binomial'):
    '''
    generate random c in {-1,1}^n or gaussian, compute <c,y> + w(y) for all y in vector_set. 
    Multiply by ln(2), so out estimator is scaled to estimate ln(Z) instead of log_2(Z)
    compute summary statistics: max, mean, stdv., median

    Inputs:
    - n: int, generate random c in {-1,1}^n
    - vector_set: list of tuples of ints (each -1 or 1).  Outer list
        is the set of vectors, each inner tuple is a particular
        vector in the set.
    - vector_weights: list of floats, weights for each vector
    - randomness: string 'binomial' or 'gaussian', if 'binomial' we draw c in {-1,1}^n
        uniformly at random as in Barvinok.  if 'gaussian' we draw each c_i from a gaussian
        with mean=0 and variance=1

    Outputs:
    - summary_statistics: dictionary with key: value pairs of
        statistic name (string): statistic value (float)
    '''
    c = []
    if randomness == 'binomial':
        #generate random c in {-1,1}^n
        for i in range(n):
            rand_c_i = np.random.rand()
            #c_i = -1
            if rand_c_i < .5: 
                c.append(-1)
            #c_i = 1
            else:
                c.append(1)
    else:
        assert(randomness == 'gaussian')
        for i in range(n):
            c.append(np.random.normal(loc=0.0, scale=1.0))
    assert(len(c) == n)

    #calculate dot product <c, y> for all y in vector_set
    def dot_product(x, y):
        return sum(x_i*y_i for x_i,y_i in zip(x, y))
    all_dot_products = []
    for y in vector_set:
        #delta estimates log_2(Z), rescale to estimate ln(Z)
        scaled_dot_product = dot_product(c,y)
        all_dot_products.append(scaled_dot_product)

    assert(len(all_dot_products)>0), all_dot_products

    summary_statistics = {'max': np.max(all_dot_products),
                          'mean': np.mean(all_dot_products),
                          'median': np.median(all_dot_products),
                          'stdv': np.std(all_dot_products)}
    return summary_statistics


def summary_stats_gumbels(n, vector_set):
    '''
    1. generate 2n gumbels with mean=0, scale=1.0 as two lists, 
    plus1_gumbels and neg1_gumbels, each of length n.
    2. find gumbel perturbation for each vector y in vector set, where
    gumbel_perturbation(y) = sum_i=1^n ( plus1_gumbels[i] if y[i] == 1
                                       + neg1_gumbels[i] if y[i] == -1 )
    3.compute summary statistics of gumbel perturbations: max, mean, stdv., median

    Inputs:
    - n: int, generate random c in {-1,1}^n
    - vector_set: list of tuples of ints (each -1 or 1).  Outer list
        is the set of vectors, each inner tuple is a particular
        vector in the set.

    Outputs:
    - summary_statistics: dictionary with key: value pairs of
        statistic name (string): statistic value (float)
    '''
    #1. generate gumbel perturbations
    plus1_gumbels = np.random.gumbel(loc=-np.euler_gamma, scale=1.0, size=n) 
    neg1_gumbels = np.random.gumbel(loc=-np.euler_gamma, scale=1.0, size=n) 

    #2. calculate gumbel perturbation for all y in vector_set
    def gumbel_perturbation(y, plus1_gumbels, neg1_gumbels):
        assert(len(y) == len(plus1_gumbels) and len(y) == len(neg1_gumbels))
        gumb_perturb = 0.0
        for (i, y_i) in enumerate(y):
            if y_i == 1:
                gumb_perturb += plus1_gumbels[i]
            else:
                assert(y_i == -1)
                gumb_perturb += neg1_gumbels[i]
        return gumb_perturb

    all_gumbel_perturbations = []
    for y in vector_set:
        all_gumbel_perturbations.append(gumbel_perturbation(y, plus1_gumbels, neg1_gumbels))

    summary_statistics = {'max': np.max(all_gumbel_perturbations),
                          'mean': np.mean(all_gumbel_perturbations),
                          'median': np.median(all_gumbel_perturbations),
                          'stdv': np.std(all_gumbel_perturbations)}
    return summary_statistics

def plot_histogram(data, statistic_type, log_Z, n, trials):
    # the histogram of the data
    plt.hist(data, bins=50)

    plt.xlabel(statistic_type)
    plt.ylabel('count')
    plt.title("%s, log(Z) = %f, n = %d, trials = %d" % (statistic_type, log_Z, n, trials))
    plt.grid(True)

#    fig.savefig('cur_test', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig("./test_bias_plots/%s_log(Z)=%f_n=%d_trials=%d.png" % (statistic_type, log_Z, n, trials))
    plt.close()   

def plot_histogram2(barvinok_data, gumbel_data, statistic_type, log_Z, n, trials, random):
    # the histogram of the data
    plt.hist(barvinok_data, bins=50, alpha=0.5, label='{-1,1} uniform bernoulli randomness')
    plt.hist(gumbel_data, bins=50, alpha=0.5, label='gumbel randomness')

    plt.xlabel(statistic_type)
    plt.ylabel('count')
    plt.title("%s, log(Z) = %f, n = %d, trials = %d" % (statistic_type, log_Z, n, trials))
    plt.grid(True)
    plt.legend(loc='upper right')

#    fig.savefig('cur_test', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig("./test_bias_plots/%s_log(Z)=%f_n=%d_trials=%d_random=%s.png" % (statistic_type, log_Z, n, trials, random))
    plt.close()   


def plot_summary_stats(n, Z, trials, random):
    '''
    Inputs:
    - n: int, generate set of Z vectors in {-1,1}^n
    - Z: int, number of vectors in set, partition function of unweighted problem
    '''
    gumbel_maxs = []
    gumbel_medians = []
    gumbel_means = []
    gumbel_stdvs = []

    barvinok_maxs = []
    barvinok_medians = []
    barvinok_means = []
    barvinok_stdvs = []

    for idx in range(trials):
        print idx/trials, "percent done"
        vector_set = generate_set(m=Z, n=n, random=random)
        barvinok_stats = summary_stats_random_c(n, vector_set, randomness='binomial')
        gumbel_stats = summary_stats_gumbels(n, vector_set)

        gumbel_maxs.append(gumbel_stats["max"])
        gumbel_medians.append(gumbel_stats["median"])
        gumbel_means.append(gumbel_stats["mean"])
        gumbel_stdvs.append(gumbel_stats["stdv"])
    
        barvinok_maxs.append(barvinok_stats["max"])
        barvinok_medians.append(barvinok_stats["median"])
        barvinok_means.append(barvinok_stats["mean"])
        barvinok_stdvs.append(barvinok_stats["stdv"])

#    plot_histogram(gumbel_maxs, "gumbel_max", math.log(Z), n, trials)
#    plot_histogram(gumbel_medians, "gumbel_median", math.log(Z), n, trials)
#    plot_histogram(gumbel_means, "gumbel_mean", math.log(Z), n, trials)
#    plot_histogram(gumbel_stdvs, "gumbel_stdv", math.log(Z), n, trials)
#
    plot_histogram(barvinok_maxs, "barvinok_max", math.log(Z, 2), n, trials)
    plot_histogram(barvinok_medians, "barvinok_median", math.log(Z, 2), n, trials)
    plot_histogram(barvinok_means, "barvinok_mean", math.log(Z, 2), n, trials)
    plot_histogram(barvinok_stdvs, "barvinok_stdv", math.log(Z, 2), n, trials)

####    plot_histogram2(barvinok_maxs, gumbel_maxs, "max", math.log(Z), n, trials, random)
####    plot_histogram2(barvinok_medians, gumbel_medians, "median", math.log(Z), n, trials, random)
####    plot_histogram2(barvinok_means, gumbel_means, "mean", math.log(Z), n, trials, random)
####    plot_histogram2(barvinok_stdvs, gumbel_stdvs, "stdv", math.log(Z), n, trials, random)

    print "ln(Z) =", math.log(Z, 2)
    print "gumbel_upper_bound =", np.mean(gumbel_maxs)
    print "barvinok estimator =", np.mean(barvinok_maxs)

def plot_estimators_vary_Z(n, trials=100, random=True):
    '''
    See how estimators compare when changing the value of Z with fixed n
    Inputs:
    - n: int, generate set of Z vectors in {-1,1}^n
    - trials: int, solve max problem trials times and take the mean
    - random: bool, true->generate random set, false->generate hypercube set
    '''
    Z_values = [2**i for i in range(n+1)]
    print Z_values
    gumbel_estimators = []
    barvinok_estimators = []
    barvinok_uppers = []
    barvinok_lowers = []

    barvinok_estimators_gaussian = []
    barvinok_gaussian_uppers = []
    barvinok_gaussian_lowers = []

    exact_log_Zs = []
    for Z in Z_values:
        print "working on Z =", Z
        gumbel_maxs = []
        barvinok_maxs = []
        barvinok_maxs_gaus = []
        for idx in range(trials):
            vector_set = generate_set(m=Z, n=n, random=random)
            barvinok_stats = summary_stats_random_c(n, vector_set, randomness='binomial')
            barvinok_stats_gaus = summary_stats_random_c(n, vector_set, randomness='gaussian')
            gumbel_stats = summary_stats_gumbels(n, vector_set)
            gumbel_maxs.append(gumbel_stats["max"])    
            barvinok_maxs.append(barvinok_stats["max"])
            barvinok_maxs_gaus.append(barvinok_stats_gaus["max"])

        gumbel_estimators.append(np.mean(gumbel_maxs))
        barvinok_estimators.append(np.mean(barvinok_maxs))
        cur_barv_est_log_2_Z = np.mean(barvinok_maxs)
        cur_barv_upper_bound = gen_upper_bound_Z(delta_bar=cur_barv_est_log_2_Z, n=n, k=trials, log_base=np.e, w_max=1, w_min=1, verbose=True)
        cur_barv_lower_bound = gen_lower_bound_Z(delta_bar=cur_barv_est_log_2_Z, n=n, k=trials, log_base=np.e, w_min=1, verbose=True)
        barvinok_uppers.append(cur_barv_upper_bound)
        barvinok_lowers.append(cur_barv_lower_bound)

        barvinok_estimators_gaussian.append(np.mean(barvinok_maxs_gaus))
        cur_gaus_barv_est_log_2_Z = np.mean(barvinok_maxs_gaus)
        cur_gaus_barv_upper_bound = gen_upper_bound_Z(delta_bar=cur_gaus_barv_est_log_2_Z, n=n, k=trials, log_base=np.e, w_max=1, w_min=1, verbose=True)
        cur_gaus_barv_lower_bound = gen_lower_bound_Z(delta_bar=cur_gaus_barv_est_log_2_Z, n=n, k=trials, log_base=np.e, w_min=1, verbose=True)
        barvinok_gaussian_uppers.append(cur_gaus_barv_upper_bound)
        barvinok_gaussian_lowers.append(cur_gaus_barv_lower_bound)

        exact_log_Zs.append(math.log(Z, 2))

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(exact_log_Zs, barvinok_estimators, 'bx', label='our estimator c in {-1,1}^n', markersize=10)
    ax.plot(exact_log_Zs, barvinok_uppers, 'b+', label='our upper c in {-1,1}^n', markersize=10)
    ax.plot(exact_log_Zs, barvinok_lowers, 'b^', label='our lower c in {-1,1}^n', markersize=7)
    ax.plot(exact_log_Zs, barvinok_estimators_gaussian, 'yx', label='our estimator c gaussian', markersize=10)
    ax.plot(exact_log_Zs, barvinok_gaussian_uppers, 'y+', label='our upper c gaussian', markersize=10)
    ax.plot(exact_log_Zs, barvinok_gaussian_lowers, 'y^', label='our lower c gaussian', markersize=7)
    ax.plot(exact_log_Zs, gumbel_estimators, 'r+', label='gumbel upper bound', markersize=10)
    ax.plot(exact_log_Zs, exact_log_Zs, 'gx', label='exact ln(Z)', markersize=10)

    ax.plot(exact_log_Zs, [val*128 for val in barvinok_lowers], 'bs', label='conjectured upper for c in {-1,1}^n', markersize=7)


    plt.title('Gumbel UB vs. Barvinok estimator, random=%s, mean over %d trials' % (random, trials))
    plt.xlabel('ln(exact Z)')
    plt.ylabel('ln(Z) or estimator')
    #make the font bigger
    matplotlib.rcParams.update({'font.size': 15})

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)

    if random:
        fig.savefig('varyZ_randomSet', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    else:
        fig.savefig('varyZ_hypercubeSet', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.close()   


def plot_estimators_vary_n(n_min ,n_max, Z, trials=100, random=True, plot_gaus_rand=False):
    '''
    See how estimators compare when changing n with fixed Z
    Inputs:
    - Z: int, number of vectors in set, partition function of unweighted problem    
    - n_min: int, min value of n to test    
    - n_max: int, max value of n to test    
    - trials: int, solve max problem trials times and take the mean
    - random: bool, true->generate random set, false->generate hypercube set
    - plot_gaus_rand: bool, if true also plot results when each element of c
        is sampled form a zero mean gaussian with standard deviation 1
    '''

#    n_min = int(math.ceil(math.log(Z)/math.log(2)))
    n_values = [i for i in range(n_min, n_max, 1)]
    print n_values
    gumbel_estimators = []
    barvinok_estimators = []
    barvinok_uppers = []
    barvinok_lowers = []

    conjectured_barv_uppers =[]

    barvinok_estimators_gaussian = []    
    barvinok_gaussian_uppers = []
    barvinok_gaussian_lowers = []

    exact_log_Zs = []
    for n in n_values:
        print "working on n =", n
        gumbel_maxs = []
        barvinok_maxs = []
        barvinok_maxs_gaus = []

        for idx in range(trials):
            vector_set = generate_set_efficient(m=Z, n=n, random=random)
            barvinok_stats = summary_stats_random_c(n, vector_set, randomness='binomial')
            barvinok_maxs.append(barvinok_stats["max"])

            if plot_gaus_rand:
                barvinok_stats_gaus = summary_stats_random_c(n, vector_set, randomness='gaussian')
                barvinok_maxs_gaus.append(barvinok_stats_gaus["max"])

            gumbel_stats = summary_stats_gumbels(n, vector_set)
            gumbel_maxs.append(gumbel_stats["max"])

        gumbel_estimators.append(np.mean(gumbel_maxs))
        barvinok_estimators.append(np.mean(barvinok_maxs))
        cur_barv_est_log_2_Z = np.mean(barvinok_maxs)
        cur_barv_upper_bound = gen_upper_bound_Z(delta_bar=cur_barv_est_log_2_Z, n=n, k=trials, log_base=np.e, w_max=1, w_min=1, verbose=True)
        cur_barv_lower_bound = gen_lower_bound_Z(delta_bar=cur_barv_est_log_2_Z, n=n, k=trials, log_base=np.e, w_min=1, verbose=True)
        barvinok_uppers.append(cur_barv_upper_bound)
        barvinok_lowers.append(cur_barv_lower_bound)

        cur_barv_upper_bound_conj = gen_upper_bound_Z_conjecture(delta_bar=cur_barv_est_log_2_Z, n=n, k=trials, log_base=np.e)
        conjectured_barv_uppers.append(cur_barv_upper_bound_conj)

        if plot_gaus_rand:
            barvinok_estimators_gaussian.append(np.mean(barvinok_maxs_gaus))
            cur_gaus_barv_est_log_2_Z = np.mean(barvinok_maxs_gaus)
            cur_gaus_barv_upper_bound = gen_upper_bound_Z(delta_bar=cur_gaus_barv_est_log_2_Z, n=n, k=trials, log_base=np.e, w_max=1, w_min=1, verbose=True)
            cur_gaus_barv_lower_bound = gen_lower_bound_Z(delta_bar=cur_gaus_barv_est_log_2_Z, n=n, k=trials, log_base=np.e, w_min=1, verbose=True)
            barvinok_gaussian_uppers.append(cur_gaus_barv_upper_bound)
            barvinok_gaussian_lowers.append(cur_gaus_barv_lower_bound)

        exact_log_Zs.append(math.log(Z, 2))

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(n_values, barvinok_estimators, 'bx', label='our estimator c in {-1,1}^n', markersize=10)
    ax.plot(n_values, barvinok_uppers, 'b+', label='our upper c in {-1,1}^n', markersize=10)
    ax.plot(n_values, barvinok_lowers, 'b^', label='our lower c in {-1,1}^n', markersize=7)
    ax.plot(n_values, [val+np.sqrt(6*n/trials) for val in barvinok_estimators], 'm+', label='conjectured upper c in {-1,1}^n', markersize=10)

#    ax.plot(n_values, conjectured_barv_uppers, 'bs', label='conjectured upper for c in {-1,1}^n', markersize=7)
#    ax.plot(n_values, [128*val for val in barvinok_lowers], 'bs', label='conjectured upper for c in {-1,1}^n', markersize=7)
  
    if plot_gaus_rand:   
        ax.plot(n_values, barvinok_estimators_gaussian, 'yx', label='our estimator c gaussian', markersize=10)
        ax.plot(n_values, barvinok_gaussian_uppers, 'y+', label='our upper c gaussian', markersize=10)
        ax.plot(n_values, barvinok_gaussian_lowers, 'y^', label='our lower c gaussian', markersize=7)
    ax.plot(n_values, gumbel_estimators, 'r+', label='gumbel upper bound', markersize=10)
    ax.plot(n_values, exact_log_Zs, 'gx', label='exact ln(Z)', markersize=10)

    plt.title('Gumbel UB vs. Barvinok estimator, random=%s, mean over %d trials' % (random, trials))
    plt.xlabel('n')
    plt.ylabel('ln(Z) or estimator')
    #make the font bigger
    matplotlib.rcParams.update({'font.size': 15})

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)
    if random:
        fig.savefig('vary_n_randomSet', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    else:
        fig.savefig('vary_n_hypercubeSet_mean%dtrials' % trials, bbox_extra_artists=(lgd,), bbox_inches='tight')            
    plt.close()   

if __name__=="__main__":
   # print generate_set(20, 10, random=True)

#######   plot_summary_stats(n=6, Z=3, trials=1000, random=True)
########   plot_summary_stats(n=15, Z=2**10, trials=1000, random=False)
########   plot_summary_stats(n=15, Z=1000, trials=1000, random=True)
########   plot_summary_stats(n=10, Z=1000, trials=100)

#    plot_estimators_vary_Z(n=13, trials=100, random=True)

#    plot_estimators_vary_n(n_min=20, n_max=20000, Z=2**10, trials=10, random=True)
    plot_estimators_vary_n(n_min=7, n_max=20, Z=2**6, trials=10000, random=False)
