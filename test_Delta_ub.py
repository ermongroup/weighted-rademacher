#test whether Delta(w) is an upper bound on Z in the unweighted case
#by enumeration

#Answer: no it isn't.  Hamming balls seem to be the worst case (smallest Delta)
#and Delta is less than Z for e.g. a hamming ball of radius 1


from __future__ import division
import operator as op
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

from itertools import chain, combinations, product



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
        assert(np.log(m)/np.log(2) - round(np.log(m)/np.log(2)) == 0) #make sure m is a power of 2
        log_2_m = int(np.log(m)/np.log(2))
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
        assert(np.log(m)/np.log(2) - round(np.log(m)/np.log(2)) == 0) #make sure m is a power of 2
        log_2_m = int(np.log(m)/np.log(2))
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



def find_max_dot_product(c, vector_set):
    '''
    compute <c,y> for all y in vector_set and return the maximum

    Inputs:
    - c: tuple in {-1,1}^n, representing a vector {-1,1}^n
    - vector_set: list of tuples of ints (each -1 or 1).  Outer list
        is the set of vectors, each inner tuple is a particular
        vector in the set.

    Outputs:
    - max_dot_product: int, maximum <c,y> over all y in vector_set
    '''
    max_dot_product = -np.inf
    #calculate dot product <c, y> for all y in vector_set
    def calc_dot_product(x, y):
        assert(len(x) == len(y))
        return sum(x_i*y_i for x_i,y_i in zip(x, y))

    for y in vector_set:
        dot_product = calc_dot_product(c,y)
        if dot_product > max_dot_product:
            max_dot_product = dot_product

    return max_dot_product

def find_max_dotProduct_hammingBall(c, hammingBall_center, hammingBall_radius):
    '''
    implicitly compute <c,y> for all y in the specified hamming ball and return the maximum
    (where a hamming ball is all vectors within a hamming distance of hammingBall_radius of
    the vector specified by hammingBall_center)

    Inputs:
    - c: tuple in {-1,1}^n, representing a vector {-1,1}^n
    - hammingBall_center: tuple in {-1,1}^n, the center of the hamming ball
    - hammingBall_radius: integer, radius of the hamming ball

    Outputs:
    - max_dot_product: int, maximum <c,y> over all y in the hamming ball
    '''
    max_dot_product = -np.inf
    #calculate dot product <c, y> for all y in vector_set
    def calc_dot_product(x, y):
        assert(len(x) == len(y))
        return sum(x_i*y_i for x_i,y_i in zip(x, y))

    dotProduct_hammingBallCenter = calc_dot_product(c,hammingBall_center)
    #if c differs from dotProduct_hammingBallCenter by at least hammingBall_radius entries
    #we can find a vector in the hamming ball that matches c by hammingBall_radius more
    #entries than hammingBall_center 
    if dotProduct_hammingBallCenter <= len(hammingBall_center) - 2*hammingBall_radius:
        max_dot_product = dotProduct_hammingBallCenter + 2*hammingBall_radius
    #otherwise the maximum dot product is simply n
    else:
        max_dot_product = len(hammingBall_center)
    return max_dot_product


def find_min_Delta(n, m):
    '''
    find the minimum expectation Delta(w) for an unweighted function
    w(x) = 0 or 1, with sum_x w(x) = m, and x in {-1,1}^n

    Inputs:
    - n: int, state space is of size 2^n
    - m: int, set of size m

    Outputs:
    - min_Delta: float, the minimum expectation for any set of size m
        among 2^n vectors

    '''

    all_possible_vectors = [] #we're going to generate every vector in {-1,1}^n
    all_vectors_helper = [[-1, 1] for i in range(n)]
    idx = 0
    for cur_vector in product(*all_vectors_helper):
        all_possible_vectors.append(cur_vector)
    assert(len(all_possible_vectors) == 2**n)

    min_Delta = np.inf
    min_vector_set = None
    for cur_vector_set in combinations(all_possible_vectors, m):
        max_dot_products_for_all_c_vectors = []
        for cur_c in all_possible_vectors:
            max_dot_products_for_all_c_vectors.append(find_max_dot_product(cur_c, cur_vector_set))
        cur_Delta = np.mean(max_dot_products_for_all_c_vectors)
        if cur_Delta < min_Delta:
            min_Delta = cur_Delta
            min_vector_set = cur_vector_set
    print "min vector set:", min_vector_set
    print "min_Delta:", min_Delta
    return min_Delta

def get_Delta_exact(vector_set, n):
    '''
    exactly find the  expectation Delta(w) for an unweighted function vector set
    w(x) = 0 or 1, and x in {-1,1}^n by explicitly enumerating all 2^n vectors

    Inputs:
    - vector_set: list of tuples, the vector set we're calculating Delta for
    - n: int, vectors are in {-1,1}^n
    Outputs:
    - Delta: float, expected value over all c {-1,1}^n of max_{x in vector_set} <c,x>

    '''

    all_possible_vectors = [] #we're going to generate every vector in {-1,1}^n
    all_vectors_helper = [[-1, 1] for i in range(n)]
    idx = 0
    for cur_vector in product(*all_vectors_helper):
        all_possible_vectors.append(cur_vector)
    assert(len(all_possible_vectors) == 2**n)

    max_dot_products_for_all_c_vectors = []
    for cur_c in all_possible_vectors:
        max_dot_products_for_all_c_vectors.append(find_max_dot_product(cur_c, vector_set))
    Delta = np.mean(max_dot_products_for_all_c_vectors)

    return Delta

def gen_random_vec(n):
    '''
    sample a vector uniformly at random from {-1,1}^n
    '''
    random_vec = np.random.rand(n)
    for r in range(n):
        if random_vec[r] < .5:
            random_vec[r] = -1
        else:
            random_vec[r] = 1
    return random_vec



def estimate_Delta(vector_set, n, k):
    '''
    estimate and bound with probability .95 the expectation Delta(w) (that is, its unweighted Rademacher complexity)
    for an unweighted function vector set w(x) = 0 or 1, and x in {-1,1}^n by computing delta_k( w) 

    Inputs:
    - vector_set: list of tuples, the vector set we're calculating Delta for
    - n: int, vectors are in {-1,1}^n
    - k: the number of random vectors we generate (and maximization problems we solve)
    Outputs:
    - delta_bar: float, our estimate of the Rademacher complexity
    - lower_bound: float, our lower bound on the Rademacher complexity that holds with probability greater than .95
    - upper_bound: float, our upper bound on the Rademacher complexity that holds with probability greater than .95
    '''
    all_deltas = []
    for i in range(k):
        cur_c = gen_random_vec(n)
        all_deltas.append(find_max_dot_product(cur_c, vector_set))

    delta_bar = np.mean(all_deltas)
    lower_bound = delta_bar - np.sqrt(6.0*n/k)
    upper_bound = delta_bar + np.sqrt(6.0*n/k)
    

    return delta_bar, lower_bound, upper_bound

def estimate_Delta_hammingBall(hammingBall_center, hammingBall_radius, n, k, testing=False, vector_set=None):
    '''
    estimate and bound with probability .95 the expectation Delta(w) (that is, its unweighted Rademacher complexity)
    for an unweighted function (w(x) = 0 or 1, and x in {-1,1}^n by computing delta_k(w)) that is shaped like
    a Hamming ball.

    Inputs:
    - hammingBall_center: tuple in {-1,1}^n, the center of the hamming ball
    - hammingBall_radius: integer, radius of the hamming ball    
    - n: int, vectors are in {-1,1}^n
    - k: the number of random vectors we generate (and maximization problems we solve)
    - testing: boolean, if true test the implementation of find_max_dotProduct_hammingBall
    - vector_set: list of tuples, the hamming ball explicitly represented as a vector set, only provide if
        testing = True

    Outputs:
    - delta_bar: float, our estimate of the Rademacher complexity
    - lower_bound: float, our lower bound on the Rademacher complexity that holds with probability greater than .95
    - upper_bound: float, our upper bound on the Rademacher complexity that holds with probability greater than .95
    '''


    all_deltas = []
    for i in range(k):
        cur_c = gen_random_vec(n)
        cur_delta = find_max_dotProduct_hammingBall(cur_c, hammingBall_center, hammingBall_radius)
        if testing:
            check_cur_delta = find_max_dot_product(cur_c, vector_set)
            assert(check_cur_delta == cur_delta)
        all_deltas.append(cur_delta)

    delta_bar = np.mean(all_deltas)
    lower_bound = delta_bar - np.sqrt(6.0*n/k)
    upper_bound = delta_bar + np.sqrt(6.0*n/k)
    

    return delta_bar, lower_bound, upper_bound


def hamming_circle(s, n, alphabet):
    """Generate strings over alphabet whose Hamming distance from s is
    exactly n.

    >>> sorted(hamming_circle('abc', 0, 'abc'))
    ['abc']
    >>> sorted(hamming_circle('abc', 1, 'abc'))
    ['aac', 'aba', 'abb', 'acc', 'bbc', 'cbc']
    >>> sorted(hamming_circle('aaa', 2, 'ab'))
    ['abb', 'bab', 'bba']

    """
    for positions in combinations(range(len(s)), n):
        for replacements in product(range(len(alphabet) - 1), repeat=n):
            cousin = list(s)
            for p, r in zip(positions, replacements):
                if cousin[p] == alphabet[r]:
                    cousin[p] = alphabet[-1]
                else:
                    cousin[p] = alphabet[r]
            yield ''.join(cousin)

def hamming_ball(s, n, alphabet):
    """Generate strings over alphabet whose Hamming distance from s is
    less than or equal to n.

    >>> sorted(hamming_ball('abc', 0, 'abc'))
    ['abc']
    >>> sorted(hamming_ball('abc', 1, 'abc'))
    ['aac', 'aba', 'abb', 'abc', 'acc', 'bbc', 'cbc']
    >>> sorted(hamming_ball('aaa', 2, 'ab'))
    ['aaa', 'aab', 'aba', 'abb', 'baa', 'bab', 'bba']

    """
    return chain.from_iterable(hamming_circle(s, i, alphabet)
                               for i in range(n + 1))


def print_vec_group_by_m(vector, m):
    for i in range(0, len(vector), m):
        print i, vector[i:i+m]

def get_all_vecs(n):
    '''
    generate all vectors in {-1,1}^n

    Inputs:
    - n: int, length of vector to generate

    Output:
    - all_vectors: list of tuples, each tuple represents a vector
        in {-1,1}^n and the list has length 2^n

    '''
    all_vectors = [] #we're going to generate every vector in {-1,1}^n
    all_vectors_helper = [[-1, 1] for i in range(n)]
    idx = 0
    for cur_vector in product(*all_vectors_helper):
        all_vectors.append(cur_vector)
    assert(len(all_vectors) == 2**n)
    return all_vectors


def calc_dot_product(x, y):
    assert(len(x) == len(y))
    return sum(x_i*y_i for x_i,y_i in zip(x, y))

def sample_chunked_perturbations(n, m):
    '''
    sample perturbation for chunks of x, with each chunk
    having m bits (except for the last one if n isn't
    divisible by m)

    corresponds to (2^m)*(n/m) random c bits in {-1,1}

    Inputs:
    - n: int, the length of vectors x in our vector space
    - m: int, generate 2^m random bits in {-1,1} for every
        m bits of a vector x (if n is not divisible by m,
        let r = remainder(n/m), then generate 2^r bits for
        the last r bits of x)

    Output: 
    - chunked_perturbations: list of dictionaries.  randomness[i] is
        a dictionary storing perturbations for bits 
        [i*m to i*m+m) in a vector x.  Each dictionary has
        key value pairs of:
            key: tuple, the values of bits [i*m to i*m+m) in a vector x
                e.g. (-1, 1, 1, -1) for m = 4
            value: int, the perturbation associated with these values
            perturbations can take values in [-m, -m+2, ..., m],
            e.g. [-4,-2,0,2,4] for m = 4
    '''
    chunked_perturbations = []

    all_length_m_vecs = get_all_vecs(m)
    for idx in range(0, n, m):
        if n - idx < m:
            num_bits = n - idx
            all_vecs = get_all_vecs(num_bits)
        else:
            num_bits = m
            all_vecs = all_length_m_vecs
        cur_chunk_perturbations = {}
        for cur_chunk_vals in all_vecs:
            #sample random chunk uniformly at random from {-1,1}^num_bits
            random_chunk = tuple([-1 if (np.random.rand()<.5) else 1 for i in range(num_bits)])
            cur_perturbation = calc_dot_product(cur_chunk_vals, random_chunk)
            cur_chunk_perturbations[cur_chunk_vals] = cur_perturbation
        chunked_perturbations.append(cur_chunk_perturbations)

    return chunked_perturbations

def find_max_chunked_perturbation(chunked_perturbations, vector_set, n, m):
    '''
    compute chunked_perturbation for all y in vector_set and return the maximum,

    Inputs:
    - chunked_perturbations: list of dictionaries.  randomness[i] is
        a dictionary storing perturbations for bits 
        [i*m to i*m+m) in a vector x.  Each dictionary has
        key value pairs of:
            key: tuple, the values of bits [i*m to i*m+m) in a vector x
                e.g. (-1, 1, 1, -1) for m = 4
            value: int, the perturbation associated with these values
            perturbations can take values in [-m, -m+2, ..., m],
            e.g. [-4,-2,0,2,4] for m = 4
    - vector_set: list of tuples of ints (each -1 or 1).  Outer list
        is the set of vectors, each inner tuple is a particular
        vector in the set.
    - n: int, the length of vectors x in our vector space
    - m: int, length of perturbation chunks (except the last one will be
        shorter if n is not divisible by m)


    Outputs:
    - max_perturbation: int, maximum perturbation over all y in vector_set
    '''

    def calc_chunked_perturbation(chunked_perturbations, y, n, m):
        '''
        calculate perturbation for the vector y, given chunked perturbations
        Inputs:
        - chunked_perturbations: list of dictionaries.  randomness[i] is
        a dictionary storing perturbations for bits 
        [i*m to i*m+m) in a vector x.  Each dictionary has
        key value pairs of:
            key: tuple, the values of bits [i*m to i*m+m) in a vector x
                e.g. (-1, 1, 1, -1) for m = 4
            value: int, the perturbation associated with these values
            perturbations can take values in [-m, -m+2, ..., m],
            e.g. [-4,-2,0,2,4] for m = 4
        - y: tuple, vector in {-1,1}^n        
        - n: int, the length of vectors x in our vector space
        - m: int, length of perturbation chunks (except the last one will be
            shorter if n is not divisible by m)

        '''
        assert(len(y) == n)
        perturbation = 0
        chunk_idx = 0
        for vec_idx in range(0, n, m):
            y_chunk = y[vec_idx:vec_idx+m]
            cur_chunk_perturbation = chunked_perturbations[chunk_idx][y_chunk]
            perturbation += cur_chunk_perturbation
            chunk_idx+=1
        assert(chunk_idx == len(chunked_perturbations))
        return perturbation

    max_perturbation = -np.inf
    #calculate perturbation for all y in vector_set

    for y in vector_set:
        perturbation = calc_chunked_perturbation(chunked_perturbations, y, n, m)
        if perturbation > max_perturbation:
            max_perturbation = perturbation

    return max_perturbation



def nCr(n, r):
    '''
    n choose r
    https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    '''
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

if __name__=="__main__":

#   find_min_Delta(3, 4)
    n = 49
    TESTING = False
    for hamming_radius in range(0, 50):
        hammingBall_size = 0 #the number of vectors in the hamming ball
        for i in range(hamming_radius+1):
            hammingBall_size += nCr(n, i)
        if TESTING:
            vector_set_strings = hamming_ball('1'*n, hamming_radius, '01')
            vector_set = []
            for vec in vector_set_strings:
                cur_vector = [1 if(char == '1') else -1 for char in vec]
                vector_set.append(tuple(cur_vector))
            assert(len(vector_set) == hammingBall_size)
        else:
            vector_set = None
        #print vector_set
        print '-'*30
        print 'hamming radius =', hamming_radius, ' number of elements in hamming ball =', hammingBall_size, 'log_2(num_elements) =', math.log(hammingBall_size)/math.log(2)
#        unweighted_rademacher_complexity = get_Delta_exact(vector_set, n)
#        print 'exact unweighted Rademacher complexity=', unweighted_rademacher_complexity

#        estimated_rad_comp, lower_bound, upper_bound = estimate_Delta(vector_set, n, k=1000)
        estimated_rad_comp, lower_bound, upper_bound = estimate_Delta_hammingBall(hammingBall_center=tuple([1 for i in range(n)]), 
                                   hammingBall_radius=hamming_radius, n=n, k=10000, 
                                   testing=TESTING, vector_set=vector_set)
        print 'estimated unweighted Rademacher complexity=', estimated_rad_comp, 'lower bound =', lower_bound, 'upper bound =', upper_bound

#        print 'weighted Rademacher complexity, Z=1:  ', unweighted_rademacher_complexity - math.log(hammingBall_size)/np.log(2)
        print 'estimate weighted Rademacher complexity, Z=1:  ', estimated_rad_comp - math.log(hammingBall_size)/math.log(2)
    sleep(3)

    TRIALS = 100
    for m in range(1, n+1):
        deltas = []
        for i in range(TRIALS):
            chunked_perturbations = sample_chunked_perturbations(n, m)
            deltas.append(find_max_chunked_perturbation(chunked_perturbations, vector_set, n, m))
        sampled_Delta = np.mean(deltas)


        print "chunked randomness by bins of length:", m, "sampled Delta =", sampled_Delta


