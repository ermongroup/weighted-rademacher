import numpy as np
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor

from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel
import pandas as pd

#when undirected edges are added, the stored node order seems to be flipped at random.
#When factors are defined, the node order matters, so explicitly specify the order to
#avoid problems, also maybe double check this.

G = MarkovModel()
G.add_nodes_from(['x1', 'x2', 'x3'])
G.add_edges_from([('x1', 'x2'), ('x1', 'x3')])

#G.add_edges_from([('z1', 'z2'), ('z1', 'z3')])
#G.add_edges_from([('x2', 'x1'), ('x3', 'x2')])
#G.add_edges_from([('x1', 'x3'), ('x1', 'x2')])
#G.add_edges_from([('a', 'c'), ('a', 'b')])
#G.add_edge('z1', 'z2')
#G.add_edge('z1', 'z3')
for edge in G.edges():
	print edge
#phi = [DiscreteFactor(edge, cardinality=[2, 2], values=np.random.rand(4)) for edge in G.edges()]

#####phi = [DiscreteFactor(edge, cardinality=[2, 2], 
#####	   values=np.array([[1,2],
#####	   					[3,4]])) for edge in G.edges()]
#G.add_nodes_from(['z1', 'z2'])
#G.add_edges_from([('z1', 'z2')])
##phi = [DiscreteFactor(edge, cardinality=[2, 2], values=np.random.rand(4)) for edge in G.edges()]
#phi = [DiscreteFactor(edge, cardinality=[2, 2], 
#	   values=np.array([[1,1],
#	   					[1,1]])) for edge in G.edges()]
#	   values=np.array([[1,1],
#	   					[1,1]])) for edge in G.edges()]

phi = [DiscreteFactor(['x2', 'x1'], cardinality=[2, 2], 
	   values=np.array([[1,2],
	   					[3,4]])),
	   DiscreteFactor(['x3', 'x1'], cardinality=[2, 2], 
	   values=np.array([[1,2],
	   					[3,4]])),
	   DiscreteFactor(['x1'], cardinality=[2], 
	   values=np.array([2,2]))]

G.add_factors(*phi)
print "factors:", G.get_factors
print "partition function =", G.get_partition_function()


def eval_partition_func_random_glass_spin(N):
	'''
	
	Inputs:
		-N: int, generate a random NxN glass spin model

	Outputs:

	'''
	G = MarkovModel()

	#create an NxN grid of nodes
	node_names = ['x%d%d' % (r,c) for r in range(N) for c in range(N)]
	print node_names
	G.add_nodes_from(node_names)

	#add an edge between each node and its 4 neighbors, except when the
	#node is on the grid border and has fewer than 4 neighbors
	edges = []
	for r in range(N):
		for c in range(N):
			if r < N-1:
				edges.append(('x%d%d' % (r,c), 'x%d%d' % (r+1,c)))
			if c < N-1:
				edges.append(('x%d%d' % (r,c), 'x%d%d' % (r,c+1)))
	assert(len(edges) == 2*N*(N-1))
	print edges
	print "number edges =", len(edges)
	G.add_edges_from(edges)	

	all_factors = []
	#sample single variable potentials
	STRONG_LOCAL_FIELD = True
	if STRONG_LOCAL_FIELD:
		f = 1 #strong local field
	else:
		f = .1 #weak local field
	for node in node_names:
		#sample in the half open interval [-f, f), gumbel paper actually uses closed interval, shouldn't matter
		theta_i = np.random.uniform(low=-f, high=f)
		factor_vals = np.array([np.exp(-theta_i), np.exp(theta_i)])
		all_factors.append(DiscreteFactor([node], cardinality=[2], values=factor_vals))

	#sample two variable potentials
	theta_ij_max = 1.5
	for edge in edges:
		#sample in the half open interval [0, theta_ij_max)
		theta_ij = np.random.uniform(low=0.0, high=theta_ij_max)
		factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
	   						    [np.exp(-theta_ij), np.exp( theta_ij)]])
		all_factors.append(DiscreteFactor(edge, cardinality=[2,2], values=factor_vals))

	G.add_factors(*all_factors)
	#print "factors:", G.get_factors
#	partition_function_enumeration = G.get_partition_function()
	partition_function_bp = get_partition_function_BP(G)
#	print "partition function enumeration =", partition_function_enumeration
	print "partition function bp =", partition_function_bp
#	assert(partition_function_bp == partition_function_enumeration)


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
	print node_names
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
	print edges
	print "number edges =", len(edges)
	G.add_edges_from(edges)	

	all_factors = []
	#sample single variable potentials
	STRONG_LOCAL_FIELD = True
	if STRONG_LOCAL_FIELD:
		f = 1 #strong local field
	else:
		f = .1 #weak local field
	for node in node_names:
		#sample in the half open interval [-f, f), gumbel paper actually uses closed interval, shouldn't matter
		theta_i = np.random.uniform(low=-f, high=f)
		factor_vals = np.array([np.exp(-theta_i), np.exp(theta_i)])
		all_factors.append(DiscreteFactor([node], cardinality=[2], values=factor_vals))

	#sample two variable potentials
	theta_ij_max = 1.5
	for edge in edges:
		#sample in the half open interval [0, theta_ij_max)
		theta_ij = np.random.uniform(low=0.0, high=theta_ij_max)
		factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
	   						    [np.exp(-theta_ij), np.exp( theta_ij)]])
		all_factors.append(DiscreteFactor(edge, cardinality=[2,2], values=factor_vals))

	G.add_factors(*all_factors)
	#print "factors:", G.get_factors
#	partition_function_enumeration = G.get_partition_function()
	partition_function_bp = get_partition_function_BP(G)
#	print "partition function enumeration =", partition_function_enumeration
	print "partition function bp =", partition_function_bp
#	assert(partition_function_bp == partition_function_enumeration)

def get_partition_function_BP(G):
	'''
	Calculate partition function of G using belief propogation

	'''
	bp = BeliefPropagation(G)
	bp.calibrate()
	clique_beliefs = bp.get_clique_beliefs()
	partition_function = np.sum(clique_beliefs.values()[0].values)
	return partition_function


def find_MAP_state(G):
	'''
	Inputs:
	- G: MarkovModel
	'''

	bp = BeliefPropagation(G)
	bp.max_calibrate()
	clique_beliefs = bp.get_clique_beliefs()
	phi_query = bp._query(G.nodes(), operation='maximize')
	print phi_query
	return phi_query

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


def find_MAP_random_glass_spin(N):
	'''
	
	Inputs:
		-N: int, generate a random NxN glass spin model

	Outputs:

	'''
	G = MarkovModel()

	#create an NxN grid of nodes
	node_names = ['x%d%d' % (r,c) for r in range(N) for c in range(N)]
	print node_names
	G.add_nodes_from(node_names)

	#add an edge between each node and its 4 neighbors, except when the
	#node is on the grid border and has fewer than 4 neighbors
	edges = []
	for r in range(N):
		for c in range(N):
			if r < N-1:
				edges.append(('x%d%d' % (r,c), 'x%d%d' % (r+1,c)))
			if c < N-1:
				edges.append(('x%d%d' % (r,c), 'x%d%d' % (r,c+1)))
	assert(len(edges) == 2*N*(N-1))
	print edges
	print "number edges =", len(edges)
	G.add_edges_from(edges)	

	all_factors = []
	#sample single variable potentials
	STRONG_LOCAL_FIELD = True
	if STRONG_LOCAL_FIELD:
		f = 1 #strong local field
	else:
		f = .1 #weak local field
	for node in node_names:
		#sample in the half open interval [-f, f), gumbel paper actually uses closed interval, shouldn't matter
		theta_i = np.random.uniform(low=-f, high=f)
		factor_vals = np.array([np.exp(-theta_i), np.exp(theta_i)])
		all_factors.append(DiscreteFactor([node], cardinality=[2], values=factor_vals))

	#sample two variable potentials
	theta_ij_max = 1.5
	for edge in edges:
		#sample in the half open interval [0, theta_ij_max)
		theta_ij = np.random.uniform(low=0.0, high=theta_ij_max)
		factor_vals = np.array([[np.exp( theta_ij), np.exp(-theta_ij)],
	   						    [np.exp(-theta_ij), np.exp( theta_ij)]])
		all_factors.append(DiscreteFactor(edge, cardinality=[2,2], values=factor_vals))

	G.add_factors(*all_factors)

#	map_state = find_MAP_state(G)
#	print "map state =", map_state

	map_val = find_MAP_val(G)
	print "map val =", map_val

	partition_function_bp = get_partition_function_BP(G)
	print "partition function bp =", partition_function_bp

def test_find_MAP():
	print '-'*80
	G = MarkovModel()
	G.add_nodes_from(['x1', 'x2', 'x3'])
	G.add_edges_from([('x1', 'x2'), ('x1', 'x3')])
	phi = [DiscreteFactor(['x2', 'x1'], cardinality=[2, 2], 
		   values=np.array([[1.0/1,1.0/2],
		   					[1.0/3,1.0/4]])),
		   DiscreteFactor(['x3', 'x1'], cardinality=[2, 2], 
		   values=np.array([[1.0/1,1.0/2],
		   					[1.0/3,1.0/4]]))]
#		   DiscreteFactor(['x1'], cardinality=[2], 
#		   values=np.array([2,2]))]
	G.add_factors(*phi)
	print "nodes:", G.nodes()

	bp = BeliefPropagation(G)
	bp.max_calibrate()
#	bp.calibrate()
	clique_beliefs = bp.get_clique_beliefs()
	print clique_beliefs
	print clique_beliefs[('x1', 'x2')]
	print clique_beliefs[('x1', 'x3')]
#	print 'partition function should be', np.sum(clique_beliefs[('x1', 'x3')].values)
	phi_query = bp._query(['x1', 'x2', 'x3'], operation='maximize')
#	phi_query = bp._query(['x1', 'x2', 'x3'], operation='marginalize')
	print phi_query

	sleep(52)	

if __name__=="__main__":
########	values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
########	                      columns=['A', 'B', 'C', 'D', 'E'])
########	model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
########	model.fit(values)
########	inference = BeliefPropagation(model)
#########	phi_query = inference._query(['A', 'B'], operation='maximize')
########	evidence = {'A':0, 'B':1, 'C':0, 'D':0, 'E':1}
########	phi_query = inference._query(variables=[], operation='maximize', evidence=evidence)
########	print 'hi'
########	print phi_query
########	print phi_query['A']
########	print phi_query['B']

####################	print '-'*80
####################	G = MarkovModel()
####################	G.add_nodes_from(['x1', 'x2', 'x3'])
####################	G.add_edges_from([('x1', 'x2'), ('x1', 'x3')])
####################
####################	phi = [DiscreteFactor(['x2', 'x1'], cardinality=[2, 2], 
####################		   values=np.array([[1,2],
####################		   					[3,4]])),
####################		   DiscreteFactor(['x3', 'x1'], cardinality=[2, 2], 
####################		   values=np.array([[1,2],
####################		   					[3,4]]))]
#####################		   DiscreteFactor(['x1'], cardinality=[2], 
#####################		   values=np.array([2,2]))]
####################
####################	G.add_factors(*phi)
#####################	junction_tree = G.to_junction_tree()
####################	bp = BeliefPropagation(G)
####################
####################
#####################	bp.max_calibrate()
####################	bp.calibrate()
####################	clique_beliefs = bp.get_clique_beliefs()
####################	print clique_beliefs
####################	print clique_beliefs[('x1', 'x2')]
####################	print clique_beliefs[('x1', 'x3')]
####################
####################	print 'partition function should be', np.sum(clique_beliefs[('x1', 'x3')].values)
####################
#####################	phi_query = bp._query(['x1', 'x2', 'x3'], operation='maximize')
####################	phi_query = bp._query(['x1', 'x2', 'x3'], operation='marginalize')
####################	print phi_query
####################
####################	print "partition function =", G.get_partition_function()

#	test_find_MAP()

	find_MAP_random_glass_spin(7)
	sleep(3)
	eval_partition_func_random_glass_spin(6)