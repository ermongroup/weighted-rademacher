import numpy as np
from itertools import chain, combinations, product

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

    max_perturbation = -np.inf
    #calculate perturbation for all y in vector_set

    for y in vector_set:
        perturbation = calc_chunked_perturbation(chunked_perturbations, y, n, m)
        if perturbation > max_perturbation:
            max_perturbation = perturbation

    return max_perturbation
