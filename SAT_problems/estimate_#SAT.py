#Code for estimating #SAT (the number of satisfying assignments).  
#We solve a SAT problem, perturbed to be a partial maximum satisfiability problem.

#import subprocess
import commands
import numpy as np
import copy
from decimal import Decimal

import matplotlib
matplotlib.use('Agg') #prevent error running remotely
import matplotlib.pyplot as plt

#Directory the open-wbo_static executable is in
#Install wbo from http://sat.inesc-id.pt/open-wbo/installation.html
WBO_DIRECTORY = '/atlas/u/jkuck/software/open-wbo'

#Directory with the MaxHS executable
#Download from http://www.maxhs.org/downloads.html
MAX_HS_DIRECTORY = '/atlas/u/jkuck/software/MaxHS-3.0/build/release/bin'

SAT_SOLVER = "MAX_HS"
#SAT_SOLVER = "WBO" #something strange seems to happen using WBO, also seems slower than MAX_HS

class SAT_problem:
    def __init__(self, nbvar, nbclauses, clauses):
        '''
        Represent an unweighted SAT problem  

        Inputs:
        - nbvar: int, number of variables 
        - nbclauses: int, the number of clauses
        - clauses: list of strings, each string represents a clause as a sequence of non-zero
            integer numbers between -nbvar and nbvar and ends with 0. Positive numbers denote 
            the corresponding variables. Negative numbers denote the negations of the 
            corresponding variables. (as in http://www.maxhs.org/docs/wdimacs.html)
        '''
        self.nbvar = nbvar
        self.nbclauses = nbclauses
        self.clauses = clauses

    def gumbel_perturb(self):
        '''
        Turn the SAT problem into a weighted partial MaxSAT, perturbing with Gumbel noise
        '''
        #many Max-SAT solvers require that top be larger that the sum of the weights of the 
        #falsified clauses in an optimal solution (top being the sum of the weights of all 
        #soft clauses plus 1 will always suffice)
        self.top = 99 
        #list of strings, each representing a soft clause. Clauses are expressed as above, 
        #with the addition that the first number of each clause line identifies the weight of the clause
        self.soft_clauses = []

        #gumbel perturbation for each variable taking the value +1
        self.pos1_gumbel_perturbs = []
        #gumbel perturbation for each variable taking the value -1
        self.neg1_gumbel_perturbs = []
        for i in range(self.nbvar):
            cur_p1_pert = np.random.gumbel(loc=-np.euler_gamma, scale=1.0)
            cur_n1_pert = np.random.gumbel(loc=-np.euler_gamma, scale=1.0)
            self.pos1_gumbel_perturbs.append(cur_p1_pert)
            self.neg1_gumbel_perturbs.append(cur_n1_pert)
            #set weight for x_i = 1 to positive gumbel difference, leaving weight for x_i = -1 at zero              
            if cur_p1_pert > cur_n1_pert:
                if SAT_SOLVER == "MAX_HS":
                    clause_weight = cur_p1_pert - cur_n1_pert
                    cur_soft_clause = "%f %d 0\n" % (clause_weight, i+1) #SAT solver expects 1 indexed variables
                elif SAT_SOLVER == "WBO":
                    clause_weight = int(round(100*(cur_p1_pert - cur_n1_pert)))
                    cur_soft_clause = "%d %d 0\n" % (clause_weight, i+1) #SAT solver expects 1 indexed variables
                else:
                    assert(False), ("Invalid SAT_SOLVER")
                self.top += clause_weight
            #set weight for x_i = -1 to positive gumbel difference, leaving weight for x_i = 1 at zero              
            else:
                if SAT_SOLVER == "MAX_HS":
                    clause_weight =  cur_n1_pert - cur_p1_pert
                    cur_soft_clause = "%f -%d 0\n" % (clause_weight, i+1) #SAT solver expects 1 indexed variables
                elif SAT_SOLVER == "WBO":
                    clause_weight =  int(round(100*(cur_n1_pert - cur_p1_pert)))
                    cur_soft_clause = "%d -%d 0\n" % (clause_weight, i+1) #SAT solver expects 1 indexed variables
                else:
                    assert(False), ("Invalid SAT_SOLVER")       
                self.top += clause_weight
            self.soft_clauses.append(cur_soft_clause)
        assert(len(self.soft_clauses) == self.nbvar)

    def barvinok_perturb(self):
        '''
        Turn the SAT problem into a weighted partial MaxSAT, perturbing with {-1,1}^n noise
        sampled uniformly at random
        '''

        #many Max-SAT solvers require that top be larger that the sum of the weights of the 
        #falsified clauses in an optimal solution (top being the sum of the weights of all 
        #soft clauses plus 1 will always suffice)
        self.top = 99 

        #list of strings, each representing a soft clause. Clauses are expressed as above, 
        #with the addition that the first number of each clause line identifies the weight of the clause
        self.soft_clauses = []
        #store the random c vector we are perturbing with explicitly
        self.perturbation_c = []
        for i in range(self.nbvar):
            self.top += 2
            c_i = np.random.rand()
            #c_i = -1, set weight for x_i = -1 to two, leaving weight for x_i = 1 at zero
            if c_i < .5: 
                self.perturbation_c.append(-1)
                cur_soft_clause = "2 -%d 0\n" % (i+1) #SAT solver expects 1 indexed variables
            #c_i = 1, set weight for x_i = 1 to two, leaving weight for x_i = -1 at zero                
            else:
                self.perturbation_c.append(1)
                cur_soft_clause = "2 %d 0\n" % (i+1) #SAT solver expects 1 indexed variables
            self.soft_clauses.append(cur_soft_clause)
        assert(len(self.soft_clauses) == self.nbvar)


def get_delta(perturbed_SAT, max_state):
    '''
    Get the value of max_x <c,x> after finding the max_state x for the SAT problem perturbed with random c

    Inputs:
    - perturbed_SAT: type SAT_problem, must be a weighted partial MaxSAT problem, perturbed
        with a c vector in {-1,1}^n
    - max_state: list of ints, each entry is either 1 or -1. 

    Outputs:
    - delta: int, max_x <c,x> for x that are satisfying solutions to the original SAT problem

    '''
    assert(len(perturbed_SAT.perturbation_c) == len(max_state))
    delta = 0
    for i in range(len(max_state)):
        delta += perturbed_SAT.perturbation_c[i]*max_state[i]
    return delta

def get_gumbel_perturbed_max(perturbed_SAT, max_state):
    '''
    Get the value of max gumbel perturbations for the specified max_state

    Inputs:
    - perturbed_SAT: type SAT_problem, must be a weighted partial MaxSAT problem, perturbed
        with two vectors of gumbel noise, perturbed_SAT.pos1_gumbel_perturbs and perturbed_SAT.neg1_gumbel_perturbs
    - max_state: list of ints, each entry is either 1 or -1. 

    Outputs:
    - gumbel_perturb: float, sum of gumbel perturbations for max_state variable states

    '''
    assert(len(perturbed_SAT.pos1_gumbel_perturbs) == len(perturbed_SAT.neg1_gumbel_perturbs))
    assert(len(perturbed_SAT.pos1_gumbel_perturbs) == len(max_state)), (len(perturbed_SAT.pos1_gumbel_perturbs), len(max_state))
    gumbel_perturb = 0.0
    for i in range(len(max_state)):
        if max_state[i] == 1:
            gumbel_perturb += perturbed_SAT.pos1_gumbel_perturbs[i]
        else:
            assert(max_state[i] == -1)
            gumbel_perturb += perturbed_SAT.neg1_gumbel_perturbs[i]
    return gumbel_perturb

def read_SAT_problem(problem_filename):
    '''
    Read in a SAT problem specified in WDIMACS format
    (http://www.maxhs.org/docs/wdimacs.html)

    Inputs:
    - problem_filename: string, the filename of the problem in WDIMACS format

    Outputs:
    - sat_problem: type SAT_problem, the SAT problem we read from problem_filename
    '''
    check_clause_count = 0
    clauses = []
    
    f = open(problem_filename, 'r')
    for line in f:
        if line[0] == 'p': #parameters line
            params = line.split()
            assert(params[1] == 'cnf') #we should be reading an unweighted SAT problem
            nbvar = int(params[2]) #number of variables
            nbclauses = int(params[3]) #number of clauses
        elif line[0] != 'c': #line isn't commented out, should be a clause
            clauses.append(line)
            check_clause_count += 1
        else:
            assert(line[0] == 'c')
    f.close()
    
    assert(check_clause_count == nbclauses), (check_clause_count, nbclauses) #did we read the correct number of clauses?
    sat_problem = SAT_problem(nbvar, nbclauses, clauses)
    return sat_problem

def write_SAT_problem(problem_filename, sat_problem, problem_type):
    '''
    Write a SAT problem to a file in WDIMACS format
    (http://www.maxhs.org/docs/wdimacs.html)

    Inputs:
        - problem_filename: string, the filename we will write the SAT problem to
        - sat_problem: type SAT_problem, the SAT problem we will write to problem_filename
        - problem_type: 'SAT' or 'weighted_MaxSat'

    Outputs:
        None, but we will write a file. Note that if a file with problem_filename exists,
        it will be erased and overwritten.
    '''
    assert(problem_type in ['SAT', 'weighted_MaxSat'])
    f = open(problem_filename, 'w')
    #write parameters line
    if problem_type == 'SAT':
        f.write("p cnf %d %d\n" % (sat_problem.nbvar, sat_problem.nbclauses))       
    else:
        f.write("p wcnf %d %d %d\n" % (sat_problem.nbvar, sat_problem.nbclauses, sat_problem.top))

    for clause in sat_problem.clauses:
        if problem_type == 'SAT':
            f.write(clause)
        else:
            f.write("%d %s" % (sat_problem.top, clause))

    if problem_type == 'weighted_MaxSat':
        if len(sat_problem.soft_clauses) > 0:
            f.write("\n")
        for soft_clause in sat_problem.soft_clauses:
            f.write(soft_clause)

    f.close()
    return None


def solve_SAT(sat_problem):
    '''
    Call a SAT solver to solve the specified SAT problem

    Inputs:
        - sat_problem: type SAT_problem, the SAT problem to solve

    Outputs:
        - satisying_solution: list of ints, each entry is either 1 or -1. 
        satisying_solution[i] is the value that variable i takes in the 
        satisfying solution to the SAT problem (1 indicates True, -1 False)
    '''
    satisying_solution = []

    write_SAT_problem('./temp_SAT_file.txt', sat_problem, problem_type='SAT')
    if SAT_SOLVER == "WBO":
        (status, output) = commands.getstatusoutput("%s/open-wbo_static ./temp_SAT_file.txt" % WBO_DIRECTORY)
    else:
        assert(SAT_SOLVER == "MAX_HS")
        (status, output) = commands.getstatusoutput("%s/maxhs ./temp_SAT_file.txt" % MAX_HS_DIRECTORY)

    for line in output.splitlines():
        if line[0] == 'v': #find the line in the output containing variable values in the solution
            params = line.split()
            assert(len(params) == sat_problem.nbvar + 1), (len(params), sat_problem.nbvar+1)
            for i in range(1, len(params)):
                if int(params[i]) > 0:
                    satisying_solution.append(1)
                else:
                    assert(int(params[i]) < 0)
                    satisying_solution.append(-1)

    return satisying_solution

def solve_weighted_partial_MaxSAT(sat_problem):
    '''
    Call a SAT solver to solve the specified SAT problem

    Inputs:
    - sat_problem: type SAT_problem, the weighted partial MaxSAT
        problem to solve

    Outputs:
    - max_solution: list of ints, each entry is either 1 or -1. 
        max_solution[i] is the value that variable i takes in the 
        weighted partial MaxSAT solution
    '''
    max_solution = []

    write_SAT_problem('./temp_SAT_file.txt', sat_problem, problem_type='weighted_MaxSat')
    if SAT_SOLVER == "WBO":
        (status, output) = commands.getstatusoutput("%s/open-wbo_static ./temp_SAT_file.txt" % WBO_DIRECTORY)
    else:
        assert(SAT_SOLVER == "MAX_HS")
        (status, output) = commands.getstatusoutput("%s/maxhs ./temp_SAT_file.txt" % MAX_HS_DIRECTORY)

    print output
    for line in output.splitlines():
        if line[0] == 'v': #find the line in the output containing variable values in the solution
            var_solutions = line.split()
#           if len(var_solutions) != sat_problem.nbvar + 1:
#               print "something SKETCHY is happening with SAT solver, extra variable in solution"
#               var_solutions.pop() #discard last variable, again, SKETCHY
            assert(len(var_solutions) == sat_problem.nbvar + 1), (len(var_solutions), sat_problem.nbvar+1)

            for i in range(1, len(var_solutions)):
                if int(var_solutions[i]) > 0:
                    max_solution.append(1)
                else:
                    assert(int(var_solutions[i]) < 0)
                    max_solution.append(-1)

    return max_solution


def estimate_sharp_sat(sat_problem, gumbel_trials, k):
    '''
    Estimate and bound the number of satisfying assignments for the 
    specified SAT problem using gumbel and barvinok perturbations

    Inputs:
    - sat_problem: type SAT_problem, the weighted partial MaxSAT
        problem to solve
    - gumbel_trials: int, take the mean of this many max solutions to gumbel 
        perturbed problems for the gumbel upper bound
    - k: int, take the mean of k values of delta for our estimator

    Outputs:
    - estimators: dictionary with (key: values) of:
        'barv_estimator': float, our estimator with barvinok perturbations
        'barv_upper': float, our upper bound on #sat with barvinok perturbations
        'barv_lower': float, our lower bound on #sat with barvinok perturbations
        'gumbel_upper': float, upper bound on #sat with gumberl perturbations
    '''
    print 'a'
    
    barv_estimator = 0.0
    for i in range(k):
        barv_pert_sat_problem = copy.deepcopy(sat_problem)
        barv_pert_sat_problem.barvinok_perturb()
        barv_max_solution = solve_weighted_partial_MaxSAT(barv_pert_sat_problem)
        delta = get_delta(barv_pert_sat_problem, barv_max_solution)
        barv_estimator += delta
    barv_estimator /= k

    print 'b'

    gumbel_upper = 0.0
    for i in range(gumbel_trials):
        gumbel_pert_sat_problem = copy.deepcopy(sat_problem)
        gumbel_pert_sat_problem.gumbel_perturb()
        gumbel_max_solution = solve_weighted_partial_MaxSAT(gumbel_pert_sat_problem)
        max_gumbel_perturbation = get_gumbel_perturbed_max(gumbel_pert_sat_problem, gumbel_max_solution)
        gumbel_upper += max_gumbel_perturbation
    gumbel_upper /= gumbel_trials

    print 'c'

    estimators = {'barv_estimator': barv_estimator,
                  'gumbel_upper': gumbel_upper}
    return estimators

if __name__=="__main__":
    NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS = 1
    OUR_K = 1
    sat_problem = read_SAT_problem("SAT_problems_cnf/%s" % "c499.isc")
    Z = 593962746002226256572855

    #write_SAT_problem('temp_SAT_file.txt', sat_problem)
    #solve_SAT(sat_problem, problem_type='SAT')

    #dictionary with key: values of model_name: number of satisfying solutions
    model_counts = {
        "log-1.cnf" : 564153552511417968750,
        "log-2.cnf" : 32334741710,
        "log-3.cnf" : 279857462060,
        "log-4.cnf" : 23421510324076617565622131248,
        "log-5.cnf" : 724152621485436659540387630662916505600,
        "tire-1.cnf": 726440820,
        "tire-2.cnf": 738969640920,
        "tire-3.cnf": 222560409176,
        "tire-4.cnf": 103191650628000,
        "ra.cnf" : 18739277038847939886754019920358123424308469030992781557966909983211910963157763678726120154469030856807730587971859910379069462105489708001873004723798633342340521799560185957916958401869207109443355859123561156747098129524433371596461424856004227854241384374972430825095073282950873641,
        "rb.cnf" : 538812462750928282721716308734898413194103864553832956073815148020987917365241105816807625188941823391012398976196112157887068449390989186368113511090801920577156367304804512491926215360520651047719401241944845883406098779319363314639309655779343140788696509960447536163784266937815828202118895534508004061478961506257883130142920912100543747226035966976598909666696626176,
        "rc.cnf" : 7711354164502494341321469992586296215389019368675210425477306574721979716330369834365339212526497517240811559116147742518536494403648202692367307372347375124735195570200184068550084511307308544710567552927267754358098889434805018720727997741618028500536976980307056282336258719799038253686515232663203157545908220322265455973957544442057437359833452997837270984970131866759705201073382727090176,
        "sat-grid-pbl-0010.cnf" : 593962746002226256572855,
        "sat-grid-pbl-0015.cnf" : 3012964503482414330783936367006634039953704207876657607,
        "sat-grid-pbl-0020.cnf" : 505529009203800755681036389586231130590495376930061598744188591803177007627164988944437560726719,
        "sat-grid-pbl-0025.cnf" : 18051755963842913831469740173876709710679368179331841451265906666609267268260641283484985491991777996404498069889539875259775880457350865091313349862141,
        "sat-grid-pbl-0030.cnf" : 154089430409193243326541334620745040441316292191902104056995281410078040886572580007404908279631836580121336881940079952338956656599622521005239507469477568002534440349139077306892061020210022834318422387583588123648727,
        "c432.isc" : 68719476736,
        "c499.isc" : 2199023255552,
        "c880.isc" : 1152921504606846976,
        "c1355.isc" : 2199023255552,
        "c1908.isc" : 8589934592,
        "c2670.isc" : 13803492693581127574869511724554050904902217944340773110325048447598592,
#        "c7552.isc" : 205688069665150755269371147819668813122841983204197482918576128
    }

    exact_log_Zs = []
    our_estimators = []
    gumbel_upper_bounds = []

    track_progress = 0
    for model_txt_file, Z in model_counts.iteritems():
        print "starting on model:", track_progress, " name:", model_txt_file
        sat_problem = read_SAT_problem("SAT_problems_cnf/%s" % model_txt_file)
        estimators = estimate_sharp_sat(sat_problem, gumbel_trials=NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS, k=OUR_K)

        exact_log_Zs.append(Decimal(Z).ln())
        our_estimators.append(estimators['barv_estimator'])
        gumbel_upper_bounds.append(estimators['gumbel_upper'])


        #print "log(z) =", Decimal(Z).ln()#np.log(Z_sgp0010)
        #print "our estimator =", estimators['barv_estimator']
        #print "gumbel_upper bound =", estimators['gumbel_upper']

        track_progress+=1


    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(range(len(model_counts)), exact_log_Zs, 'go', label='exact log(Z)')
    ax.plot(range(len(model_counts)), our_estimators, 'r*', label='our estimator, k=%d' %OUR_K)
    ax.plot(range(len(model_counts)), gumbel_upper_bounds, 'bs', label='gumbel upper bound num_perturb=%d' % NUM_GUMBEL_UPPER_BOUND_PERTURBATIONS)


    plt.title('#SAT Model Count upper Bounds and Estimates')
    plt.xlabel('model index, not meaningful')
    plt.ylabel('log(Z)')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('cur_test', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()   
