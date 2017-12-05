from __future__ import division
import numpy as np

def lower_bound_Z(delta_bar, n, k, log_base, w_min=None, verbose=False):
    '''
    Lower bound the partition function

    Inputs:
    - delta_bar: float, our estimator of log_2(Z).  The mean of 
        max_x{<c,x> + log_2 w(x)} over k randomly generated c vectors.
    - n: int.  Our space is of size 2^n (vectors are in {-1,1}^n, maximum 
        value of the partition function is w_max*2^n)
    - k: int.  number of random c vectors used to compute delta_bar
    - log_base: float.  We will return a lower bound on log(Z) using this base
    - w_min: float, the smallest weight.  If available we may be able to improve
        the lower bound using it.
    - verbose: bool, if true print info about which bound we use
    Outputs:
    - high_prob_LB: float, lower bound on log(Z) in the specified base that holds
        with high probability.  
    - expectation_LB: float, lower bound on log(Z) in the specified base that holds
        in expecatation (if delta_bar was calculated as mean of infinite number of deltas)
    '''
    #want an estimator of log(Z) with base log_base, rescale delta

################### Calculate high probability lower bound ###################
    if w_min:
        log_w_min = np.log(w_min)/np.log(2)      
        lamda = (delta_bar - np.sqrt(6*n/k) - log_w_min)/n
        if lamda <= 1.0: 
            lower_bound_w_min = (delta_bar - np.sqrt(6*n/k) - log_w_min)**2/(2*n) + log_w_min
    lower_bound_no_w_min = delta_bar - np.sqrt(6*n/k) - n/2
    if w_min and lamda <= 1.0:
        #this should be true for lower bound using \Delta, but is it with this using \delta?
        assert(lower_bound_w_min >= lower_bound_no_w_min) 
        if verbose:
            print "high probability LOWER BOUND CALCULATION: used w_min"
        high_prob_LB = lower_bound_w_min
    else:
        if verbose:
            print "high probability LOWER BOUND CALCULATION: did not use w_min"
        high_prob_LB = lower_bound_no_w_min

################### Calculate lower bound that holds in expectation for comparison with gumbel ###################
    if w_min:
        log_w_min = np.log(w_min)/np.log(2)              
        lamda = (delta_bar - log_w_min)/n
        if lamda <= 1.0: 
            lower_bound_w_min = (delta_bar - log_w_min)**2/(2*n) + log_w_min
    lower_bound_no_w_min = delta_bar - n/2
    if w_min and lamda <= 1.0:
        #this should be true for lower bound using \Delta, but is it with this using \delta?
        assert(lower_bound_w_min >= lower_bound_no_w_min) 
        if verbose:
            print "expectation LOWER BOUND CALCULATION: used w_min"
        expectation_LB = lower_bound_w_min
    else:
        if verbose:
            print "expectation LOWER BOUND CALCULATION: did not use w_min"
        expectation_LB = lower_bound_no_w_min


    rescaled_high_prob_LB = high_prob_LB*np.log(2)/np.log(log_base)
    rescaled_expectation_LB = expectation_LB*np.log(2)/np.log(log_base)
    return (rescaled_high_prob_LB, rescaled_expectation_LB)


def OLD_WRONG_LOG_BASE_upper_bound_Z(delta_bar, n, k, log_base=np.e, w_max=None, w_min=None, verbose=False, chunk_size=None):
    '''
    Upper bound the partition function

    Inputs:
    - delta_bar: float, our estimator of log_2(Z).  The mean of 
        max_x{<c,x> + log_2 w(x)} over k randomly generated c vectors.
    - n: int.  Our space is of size 2^n (vectors are in {-1,1}^n, maximum 
        value of the patition function is w_max*2^n)
    - k: int.  number of random c vectors used to compute delta_bar
    - log_base: float.  We will return an upper bound on log(Z) using this base
        CURRENTLY, must be e
    - w_max: float, the largest weight.  If available we may be able to improve
        the lower bound using it.
    - w_min: float, the smallest weight.  If available we may be able to improve
        the lower bound using it.       
    - verbose: bool, if true print info about which bound we use
    - chunk_size: int or None, we use n*2^chunk_size bits of randomness.  None when we
        only use n bits of randomness, only worked out for chunk_size = 1

    Outputs:
    - high_prob_best_upper_bound: float, our upper bound on ln(Z) that holds with high 
        probability
    - expectation_best_upper_bound: float, our upper bound on ln(Z) that holds in 
        expectation (if delta_bar was calculated as mean of infinite number of deltas)

    '''
    assert(log_base==np.e), "haven't worked out upper bound for other log bases"
    #want an estimator of log(Z) with base log_base, rescale delta
    scaled_delta_bar = delta_bar*np.log(2)/np.log(log_base)


################### Calculate high probability upper bound ###################

    if chunk_size == None:
        if w_min:
            log_w_min = np.log(w_min)/np.log(log_base)      
            beta_w_min = (scaled_delta_bar + np.sqrt(6*n/k) - log_w_min)/n
            if 0 < beta_w_min and beta_w_min < 1/(1+np.e): #we can use beta_w_min
                upper_bound_w_min = np.log((1 - beta_w_min)/beta_w_min)*(scaled_delta_bar + np.sqrt(6*n/k) - log_w_min) \
                                  - n*np.log(1 - beta_w_min) + log_w_min
            else:
                upper_bound_w_min = np.inf
        else:
            upper_bound_w_min = np.inf            
        print "w_min =", w_min, '6'*80
    
        if w_max:
            log_w_max = np.log(w_max)/np.log(log_base)      
            beta_w_max = (scaled_delta_bar + np.sqrt(6*n/k) - log_w_max)/n
            if 1/(1+np.e) >= beta_w_max or beta_w_max > 1/2:
                beta_w_max = .5 #always can use this trivial upper bound, compute to see if it's best
            upper_bound_w_max = np.log((1 - beta_w_max)/beta_w_max)*(scaled_delta_bar + np.sqrt(6*n/k) - log_w_max) \
                              - n*np.log(1 - beta_w_max) + log_w_max
    
        upper_bound_no_weight = scaled_delta_bar + np.sqrt(6*n/k) + n*np.log(1/np.e + 1)
    
        high_prob_best_upper_bound = min([upper_bound_w_min, upper_bound_w_max, upper_bound_no_weight])
    else:
        assert(chunk_size == 1)
        if w_min:
            log_w_min = np.log(w_min)/np.log(log_base)      
            beta_w_min = 2*(scaled_delta_bar + np.sqrt(6*n/k) - log_w_min)/n
            if 0 < beta_w_min and beta_w_min < 1/(1+np.sqrt(np.e)): #we can use beta_w_min
                upper_bound_w_min = 2*np.log((1 - beta_w_min)/beta_w_min)*(scaled_delta_bar + np.sqrt(6*n/k) - log_w_min) \
                                  - n*np.log(1 - beta_w_min) + log_w_min
            else:
                upper_bound_w_min = np.inf
        else:
            upper_bound_w_min = np.inf            
        print "w_min =", w_min, '6'*80
    
        if w_max:
            log_w_max = np.log(w_max)/np.log(log_base)      
            beta_w_max = 2*(scaled_delta_bar + np.sqrt(6*n/k) - log_w_max)/n
            if 1/(1+np.sqrt(np.e)) >= beta_w_max or beta_w_max > 1/2:
                beta_w_max = .5 #always can use this trivial upper bound, compute to see if it's best
            upper_bound_w_max = 2*np.log((1 - beta_w_max)/beta_w_max)*(scaled_delta_bar + np.sqrt(6*n/k) - log_w_max) \
                              - n*np.log(1 - beta_w_max) + log_w_max
    
        upper_bound_no_weight = scaled_delta_bar + np.sqrt(6*n/k) + n*np.log(1/np.sqrt(np.e) + 1)
        assert(np.abs(upper_bound_no_weight - (scaled_delta_bar + np.sqrt(6*n/k) + n*.47408))/upper_bound_no_weight < .001), ((scaled_delta_bar + np.sqrt(6*n/k) + n*.47408), upper_bound_no_weight)
    
        high_prob_best_upper_bound = min([upper_bound_w_min, upper_bound_w_max, upper_bound_no_weight])

    if verbose:
        if high_prob_best_upper_bound == upper_bound_w_max:
            print "high probability UPPER BOUND CALCULATION: used w_max, beta =", beta_w_max
        elif high_prob_best_upper_bound == upper_bound_w_min:
            print "high probability UPPER BOUND CALCULATION: used w_min"
        else:
            assert(high_prob_best_upper_bound == upper_bound_no_weight)
            print "high probability UPPER BOUND CALCULATION: did not use w_min or w_max"

################### Calculate upper bound that holds in expectation for comparison with gumbel ###################
    if chunk_size == None:
        if w_min:
            log_w_min = np.log(w_min)/np.log(log_base)      
            beta_w_min = (scaled_delta_bar - log_w_min)/n
            if 0 < beta_w_min and beta_w_min < 1/(1+np.e): #we can use beta_w_min
                upper_bound_w_min = np.log((1 - beta_w_min)/beta_w_min)*(scaled_delta_bar - log_w_min) \
                                  - n*np.log(1 - beta_w_min) + log_w_min
            else:
                upper_bound_w_min = np.inf
        else:
            upper_bound_w_min = np.inf            
        print "w_min =", w_min, '6'*80
    
        if w_max:
            log_w_max = np.log(w_max)/np.log(log_base)      
            beta_w_max = (scaled_delta_bar - log_w_max)/n
            if 1/(1+np.e) >= beta_w_max or beta_w_max > 1/2:
                beta_w_max = .5 #always can use this trivial upper bound, compute to see if it's best
            upper_bound_w_max = np.log((1 - beta_w_max)/beta_w_max)*(scaled_delta_bar - log_w_max) \
                              - n*np.log(1 - beta_w_max) + log_w_max
    
        upper_bound_no_weight = scaled_delta_bar + n*np.log(1/np.e + 1)
    
        expectation_best_upper_bound = min([upper_bound_w_min, upper_bound_w_max, upper_bound_no_weight])
    else:
        assert(chunk_size == 1)
        if w_min:
            log_w_min = np.log(w_min)/np.log(log_base)      
            beta_w_min = 2*(scaled_delta_bar - log_w_min)/n
            if 0 < beta_w_min and beta_w_min < 1/(1+np.sqrt(np.e)): #we can use beta_w_min
                upper_bound_w_min = 2*np.log((1 - beta_w_min)/beta_w_min)*(scaled_delta_bar - log_w_min) \
                                  - n*np.log(1 - beta_w_min) + log_w_min
            else:
                upper_bound_w_min = np.inf
        else:
            upper_bound_w_min = np.inf            
        print "w_min =", w_min, '6'*80
    
        if w_max:
            log_w_max = np.log(w_max)/np.log(log_base)      
            beta_w_max = 2*(scaled_delta_bar - log_w_max)/n
            if 1/(1+np.sqrt(np.e)) >= beta_w_max or beta_w_max > 1/2:
                beta_w_max = .5 #always can use this trivial upper bound, compute to see if it's best
            upper_bound_w_max = 2*np.log((1 - beta_w_max)/beta_w_max)*(scaled_delta_bar - log_w_max) \
                              - n*np.log(1 - beta_w_max) + log_w_max
    
        upper_bound_no_weight = scaled_delta_bar + n*np.log(1/np.sqrt(np.e) + 1)
        assert(np.abs(upper_bound_no_weight - (scaled_delta_bar + n*.47408))/upper_bound_no_weight < .001), ((scaled_delta_bar + n*.47408), upper_bound_no_weight)
    
        expectation_best_upper_bound = min([upper_bound_w_min, upper_bound_w_max, upper_bound_no_weight])

    if verbose:
        if expectation_best_upper_bound == upper_bound_w_max:
            print "expectation UPPER BOUND CALCULATION: used w_max, beta =", beta_w_max
        elif expectation_best_upper_bound == upper_bound_w_min:
            print "expectation UPPER BOUND CALCULATION: used w_min"
        else:
            assert(expectation_best_upper_bound == upper_bound_no_weight)
            print "expectation UPPER BOUND CALCULATION: did not use w_min or w_max"

    return (high_prob_best_upper_bound, expectation_best_upper_bound)




def upper_bound_Z(delta_bar, n, k, log_base=np.e, w_max=None, w_min=None, verbose=False):
    '''
    Upper bound the partition function

    Inputs:
    - delta_bar: float, our estimator of log_2(Z).  The mean of 
        max_x{<c,x> + log_2 w(x)} over k randomly generated c vectors.
    - n: int.  Our space is of size 2^n (vectors are in {-1,1}^n, maximum 
        value of the patition function is w_max*2^n)
    - k: int.  number of random c vectors used to compute delta_bar
    - log_base: float.  We will return an upper bound on log(Z) using this base
        CURRENTLY, must be e
    - w_max: float, the largest weight.  If available we may be able to improve
        the lower bound using it.
    - w_min: float, the smallest weight.  If available we may be able to improve
        the lower bound using it.       
    - verbose: bool, if true print info about which bound we use

    Outputs:
    - high_prob_best_upper_bound: float, our upper bound on ln(Z) that holds with high 
        probability
    - expectation_best_upper_bound: float, our upper bound on ln(Z) that holds in 
        expectation (if delta_bar was calculated as mean of infinite number of deltas)

    '''
################### Calculate high probability upper bound ###################

    #based on beta without enumerating:

    high_prob_best_upper_bound = None
    if w_min:
        log_w_min = np.log(w_min)/np.log(2)      
        beta_w_min = (delta_bar + np.sqrt(6*n/k) - log_w_min)/n
        if 0 < beta_w_min and beta_w_min < 1/(3): #we can use beta_w_min
            high_prob_best_upper_bound = np.log((1 - beta_w_min)/beta_w_min)/np.log(2)*(delta_bar + np.sqrt(6*n/k) - log_w_min) \
                              - n*np.log(1 - beta_w_min)/np.log(2) + log_w_min
    if high_prob_best_upper_bound == None and w_max:        
        log_w_max = np.log(w_max)/np.log(2)
        beta_w_max = (delta_bar + np.sqrt(6*n/k) - log_w_max)/n
        if beta_w_max > 1/2:
            beta_w_max = 1/2
        if 1/3 <= beta_w_max:
            high_prob_best_upper_bound = np.log((1 - beta_w_max)/beta_w_max)/np.log(2)*(delta_bar + np.sqrt(6*n/k) - log_w_max) \
                          - n*np.log(1 - beta_w_max)/np.log(2) + log_w_max

    if high_prob_best_upper_bound == None:        
        high_prob_best_upper_bound = delta_bar + np.sqrt(6*n/k) + n*np.log(3/2)/np.log(2)
    

    ##############TESTING##############
    if w_min:
        log_w_min = np.log(w_min)/np.log(2)      
        beta_w_min = (delta_bar + np.sqrt(6*n/k) - log_w_min)/n
        if 0 < beta_w_min and beta_w_min < 1/(3): #we can use beta_w_min
            upper_bound_w_min = np.log((1 - beta_w_min)/beta_w_min)/np.log(2)*(delta_bar + np.sqrt(6*n/k) - log_w_min) \
                              - n*np.log(1 - beta_w_min)/np.log(2) + log_w_min
        else:
            upper_bound_w_min = np.inf
    else:
        upper_bound_w_min = np.inf            
    print "w_min =", w_min, '6'*80
    
    if w_max:
        log_w_max = np.log(w_max)/np.log(2)
        beta_w_max = (delta_bar + np.sqrt(6*n/k) - log_w_max)/n
        if 1/3 >= beta_w_max or beta_w_max > 1/2:
            beta_w_max = .5 #always can use this trivial upper bound, compute to see if it's best
        upper_bound_w_max = np.log((1 - beta_w_max)/beta_w_max)/np.log(2)*(delta_bar + np.sqrt(6*n/k) - log_w_max) \
                          - n*np.log(1 - beta_w_max)/np.log(2) + log_w_max

    
    upper_bound_no_weight = delta_bar + np.sqrt(6*n/k) + n*np.log(3/2)/np.log(2)
    
    check_high_prob_best_upper_bound = min([upper_bound_w_min, upper_bound_w_max, upper_bound_no_weight])
    assert(np.abs(check_high_prob_best_upper_bound -high_prob_best_upper_bound) < .0001), (check_high_prob_best_upper_bound, high_prob_best_upper_bound, delta_bar, upper_bound_w_min, upper_bound_w_max, upper_bound_no_weight, w_min, w_max)

    ##############DONE TESTING##############

    if verbose:
        if high_prob_best_upper_bound == upper_bound_w_max:
            print "high probability UPPER BOUND CALCULATION: used w_max, beta =", beta_w_max
            print "upper_bound_w_max =", np.log((1 - beta_w_max)/beta_w_max)/np.log(2)*(delta_bar + np.sqrt(6*n/k) - log_w_max) \
                                         - n*np.log(1 - beta_w_max)/np.log(2) + log_w_max
            print "upper_bound_w_max =", np.log((1 - beta_w_max)/beta_w_max)/np.log(2)*(delta_bar + np.sqrt(6*n/k) - log_w_max)
            print "upper_bound_w_max =", np.log((1 - beta_w_max)/beta_w_max)/np.log(2)
            print "upper_bound_w_max =", (1 - beta_w_max)/beta_w_max
            print 'beta_w_max =', beta_w_max            
            print "upper_bound_w_max =", (delta_bar + np.sqrt(6*n/k) - log_w_max)

        elif high_prob_best_upper_bound == upper_bound_w_min:
            print "high probability UPPER BOUND CALCULATION: used w_min"
        else:
            assert(high_prob_best_upper_bound == upper_bound_no_weight)
            print "high probability UPPER BOUND CALCULATION: did not use w_min or w_max"

################### Calculate upper bound that holds in expectation for comparison with gumbel ###################
    #based on beta without enumerating:

    expectation_best_upper_bound = None
    if w_min:
        log_w_min = np.log(w_min)/np.log(2)      
        beta_w_min = (delta_bar - log_w_min)/n
        if 0 < beta_w_min and beta_w_min < 1/(3): #we can use beta_w_min
            expectation_best_upper_bound = np.log((1 - beta_w_min)/beta_w_min)/np.log(2)*(delta_bar - log_w_min) \
                              - n*np.log(1 - beta_w_min)/np.log(2) + log_w_min
    if expectation_best_upper_bound == None and w_max:        
        log_w_max = np.log(w_max)/np.log(2)
        beta_w_max = (delta_bar - log_w_max)/n
        if beta_w_max > 1/2:
            beta_w_max = 1/2
        if 1/3 <= beta_w_max:
            expectation_best_upper_bound = np.log((1 - beta_w_max)/beta_w_max)/np.log(2)*(delta_bar - log_w_max) \
                          - n*np.log(1 - beta_w_max)/np.log(2) + log_w_max

    if expectation_best_upper_bound == None:        
        expectation_best_upper_bound = delta_bar + n*np.log(3/2)/np.log(2)

    ##############TESTING##############

    if w_min:
        log_w_min = np.log(w_min)/np.log(2)
        beta_w_min = (delta_bar - log_w_min)/n
        if 0 < beta_w_min and beta_w_min < 1/3: #we can use beta_w_min
            upper_bound_w_min = np.log((1 - beta_w_min)/beta_w_min)/np.log(2)*(delta_bar - log_w_min) \
                              - n*np.log(1 - beta_w_min)/np.log(2) + log_w_min
        else:
            upper_bound_w_min = np.inf
    else:
        upper_bound_w_min = np.inf            
    print "w_min =", w_min, '6'*80
    
    if w_max:
        log_w_max = np.log(w_max)/np.log(2)
        beta_w_max = (delta_bar - log_w_max)/n
        if 1/3 >= beta_w_max or beta_w_max > 1/2:
            beta_w_max = .5 #always can use this trivial upper bound, compute to see if it's best
        upper_bound_w_max = np.log((1 - beta_w_max)/beta_w_max)/np.log(2)*(delta_bar - log_w_max) \
                          - n*np.log(1 - beta_w_max)/np.log(2) + log_w_max
    
    upper_bound_no_weight = delta_bar + n*np.log(3/2)/np.log(2)
    
    check_expectation_best_upper_bound = min([upper_bound_w_min, upper_bound_w_max, upper_bound_no_weight])
    assert(np.abs(check_expectation_best_upper_bound - expectation_best_upper_bound) < .0001), (check_expectation_best_upper_bound, expectation_best_upper_bound, delta_bar, upper_bound_w_min, upper_bound_w_max, upper_bound_no_weight, w_min, w_max)

    ##############DONE TESTING##############




    if verbose:
        if expectation_best_upper_bound == upper_bound_w_max:
            print "expectation UPPER BOUND CALCULATION: used w_max, beta =", beta_w_max
            print "upper_bound_w_max = ", np.log((1 - beta_w_max)/beta_w_max)/np.log(2)*(delta_bar - log_w_max) \
                                            - n*np.log(1 - beta_w_max)/np.log(2) + log_w_max
            print "upper_bound_w_max = ", np.log((1 - beta_w_max)/beta_w_max)/np.log(2)*(delta_bar - log_w_max) 
            print "upper_bound_w_max = ", np.log((1 - beta_w_max)/beta_w_max)
            print "upper_bound_w_max = ", (1 - beta_w_max)/beta_w_max 
            print 'beta_w_max =', beta_w_max
            print "upper_bound_w_max = ", (delta_bar - log_w_max) 

        elif expectation_best_upper_bound == upper_bound_w_min:
            print "expectation UPPER BOUND CALCULATION: used w_min"
        else:
            assert(expectation_best_upper_bound == upper_bound_no_weight)
            print "expectation UPPER BOUND CALCULATION: did not use w_min or w_max"

    rescaled_high_prob_best_upper_bound = high_prob_best_upper_bound*np.log(2)/np.log(log_base)
    rescaled_expectation_best_upper_bound = expectation_best_upper_bound*np.log(2)/np.log(log_base)

    print '#'*10, "high_prob_best_upper_bound", high_prob_best_upper_bound
    print '#'*10, "expectation_best_upper_bound", expectation_best_upper_bound

#    return (high_prob_best_upper_bound, expectation_best_upper_bound)
    return (rescaled_high_prob_best_upper_bound, rescaled_expectation_best_upper_bound)



def calculate_gumbel_slack(A, M, delta):
    '''
    Calculate the slack between the Gumbel upper bound that holds in expectation and
    the bound that holds with high probability, equations on page 32 of:
    https://arxiv.org/pdf/1602.03571.pdf

    Inputs:
    - A: int, the set size |A|, perturbing each variable individually this should be
        n, the number of variables
    - M: int, we take the mean of M solutions to independently perturbed maximization
        problems
    - delta: float, the upper bound will hold with probability 1-delta (NOTE: this
        delta is from the Gumbel paper mentioned mentioned above and is unrelated
        to the delta from our paper which is elsewhere in this code)

    Outputs:
    - best_slack: float, the tightest slack between the two options
    '''
    slack1 = 2*np.sqrt(A)*(1 + np.sqrt(1/(2*M)*np.log(2/delta)))**2
#    print "slack1 =", slack1
    slack2 = np.sqrt(A)*max(4/M * np.log(2/delta),
                            np.sqrt(32/M * np.log(2/delta)))
#    print "slack2 =", slack2
#    print "slack2a =", np.sqrt(A)*4/M * np.log(2/delta)
#    print "slack2b =", np.sqrt(A)*np.sqrt(32/M * np.log(2/delta))

    best_slack = min(slack1, slack2)
    return best_slack 

def upper_bound_Z_conjecture(delta_bar, n, k, log_base=np.e):
    '''
    test Stefano's conjectured upper bound on the partition function

    Inputs:
    - delta_bar: float, our estimator of log_2(Z).  The mean of 
        max_x{<c,x> + log_2 w(x)} over k randomly generated c vectors.
    - n: int.  Our space is of size 2^n (vectors are in {-1,1}^n, maximum 
        value of the patition function is w_max*2^n)
    - k: int.  number of random c vectors used to compute delta_bar
    - log_base: float.  We will return an upper bound on log(Z) using this base
        CURRENTLY, must be e
    - w_max: float, the largest weight.  If available we may be able to improve
        the lower bound using it.
    - w_min: float, the smallest weight.  If available we may be able to improve
        the lower bound using it.       
    - verbose: bool, if true print info about which bound we use

    Outputs:
    - conjectured_upper_bound: float, conjectured upper bound on ln(Z) 
    '''
    scaled_delta_bar = delta_bar*np.log(2)/np.log(log_base)
#    conjectured_upper_bound = 64*(scaled_delta_bar + np.sqrt(6*n/k))**2/n 
    conjectured_upper_bound = 64*(scaled_delta_bar)**2/n 
#    conjectured_upper_bound = 64*(scaled_delta_bar - np.sqrt(6*n/k))**2/n 

    return conjectured_upper_bound


if __name__ == "__main__":
    print calculate_gumbel_slack(A=1000, M=10, delta=.05)
