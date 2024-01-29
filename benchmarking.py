from time import process_time_ns
from numpy import array
from mmaformatter import save_as_mma

import our_method
import their_method



'''
    Convenience functions
'''

def getRandomMatrix(numQubits):
    from random import random as r
    dim = 1 << numQubits
    matrix = [[complex(r(), r()) for _ in range(dim)] for _ in range(dim)]
    matrix = array([array(r) for r in matrix])
    return matrix

def repeatedlyTime(func, numQubits, numReps):
    durs = []

    for _ in range(numReps):
        matrix = getRandomMatrix(numQubits)

        t1 = process_time_ns()
        func(matrix)
        t2 = process_time_ns()

        durs.append(t2 - t1)
    
    return durs



'''
    Testing correctness
'''

for numQubits in range(1, 5):
    matrix = getRandomMatrix(numQubits)

    our_coeffs = our_method.get_all_coefficients(matrix)
    their_coeffs = their_method.get_all_coefficients(matrix)

    for c1, c2 in zip(our_coeffs, their_coeffs):
        assert abs(c1-c2) < 1E-10

print("All tests passed\n")



'''
    Benchmarking
'''

# config
numReps = 100
minQubits = 1
maxQubits = 8

# output data
outFn = 'results.txt'
results = {
    'numReps':numReps, 
    'minQubits': minQubits,
    'maxQubits': maxQubits,
    'reachedQubits': 0,
    'durations': {}
}

print(f"Beginning benchmarks with {numReps} repetitions from 1 to {maxQubits} qubits.")

for numQubits in range(minQubits, maxQubits+1):
    print(f'\nnumQubits = {numQubits}...')

    # repeatedly time both functions
    ourDurs = repeatedlyTime(our_method.get_all_coefficients, numQubits, numReps)
    theirDurs = repeatedlyTime(their_method.get_all_coefficients, numQubits, numReps)

    # save all results so far to file (overwriting, so job can be safely interrupted)
    results['durations'][numQubits] = {'ours': ourDurs, 'theirs': theirDurs}
    save_as_mma(results, outFn)

    # give the user a sneak peek
    print('\tours:', sum(ourDurs)/float(numReps), 'ns')
    print('\ttheirs:', sum(theirDurs)/float(numReps), 'ns')
    print('\tratio:', sum(ourDurs)/float(sum(theirDurs)))