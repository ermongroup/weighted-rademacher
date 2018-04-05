import numpy as np
import math
import matplotlib
matplotlib.use('Agg') #prevent error running remotely
import matplotlib.pyplot as plt

def estimate_z_iid_c(n, set_size, trials):
    '''
    Estimate the size of set containing set_size vectors in {-1,1}^n
    by 

    1.generating a random vector c_j in {-1,1}^n for each vector
    x in the set
    2.taking the dot product between each vector x in the set and
    its random c
    3.using the maximum dot product as an estimator of sqrt(log(set_size))

    Inputs:
    - n: int, the set contains vectors of length n (in {-1,1}^n)
    - set_size: estimate the size of a set with set_size vectors
    - trials: int, take the mean of trials independent estimators

    Output:
    - mean_estimator: an estimator of log(set_size)
    '''

    max_dot_products = []
    for i in range(trials):
        max_dot_product = np.max(np.random.binomial(n=n, p=.5, size=set_size))*2-n
#        max_dot_product = (np.max(np.random.binomial(n=n, p=.5, size=set_size)))
        max_dot_products.append(max_dot_product)
    mean_estimator = np.mean(max_dot_products)
    return mean_estimator


def plot_vary_Z_iid_c(n, trials):
    '''
    Plot our sstimates of the size of a set containing 
    set_size vectors in {-1,1}^n for various set sizes
    by 

    1.generating a random vector c_j in {-1,1}^n for each vector
    x in the set
    2.taking the dot product between each vector x in the set and
    its random c
    3.using the maximum dot product as an estimator of sqrt(log(set_size))

    Inputs:
    - n: int, the set contains vectors of length n (in {-1,1}^n)
    - trials: int, take the mean of trials independent estimators

    '''

    set_sizes = [2**i for i in range(n)]

    estimators = []
    delta_bars = []
    log_set_sizes = []

    for set_size in set_sizes:
        delta_bar = estimate_z_iid_c(n,set_size,trials)
        delta_bars.append(delta_bar)
        cur_est = (delta_bar**2)/n
        estimators.append(cur_est)
        log_set_sizes.append(math.log(set_size, 2))
#        log_set_sizes.append(math.log(set_size, np.e))
        print "log set size =", log_set_sizes[-1]
        print "our estimator =", estimators[-1]


    fig = plt.figure()
    ax = plt.subplot(111)


    ax.plot(log_set_sizes, delta_bars, 'b+', label='delta_bar, c iid, average over %d trials' % trials, markersize=10)
    ax.plot(log_set_sizes, estimators, 'r+', label='(delta_bar^2)/n c iid, average over %d trials' % trials, markersize=10)
    ax.plot(log_set_sizes, log_set_sizes, 'gx', label='log_2(Z)', markersize=10)

    plt.title('iid c estimator')
    plt.xlabel('log_2(Z)')
    plt.ylabel('log_2(Z) or estimator')
    #make the font bigger
    matplotlib.rcParams.update({'font.size': 15})

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)
    fig.savefig('iid_c_estimator', bbox_extra_artists=(lgd,), bbox_inches='tight')            
    plt.close()   

print estimate_z_iid_c(n=15, set_size=16, trials=100)
#sleep(2)

plot_vary_Z_iid_c(20, 100)