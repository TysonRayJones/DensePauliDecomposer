import sys
from time import process_time_ns
from numpy import array

from utils.mmaformatter import save_as_mma

import methods.jones
import methods.romero_et_al
import methods.hamaguchi_et_al
import methods.hantzko_et_al



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

print('Testing correctnesss (and invoking JIT)...')

for numQubits in range(1, 5):
    print('numQubits =', numQubits)

    matrix = getRandomMatrix(numQubits)
    
    jone_coeffs = methods.jones.calcPauliVector(matrix)
    rome_coeffs = methods.romero_et_al.calcPauliVector(matrix)
    hama_coeffs = methods.hamaguchi_et_al.calcPauliVector(matrix)
    hant_coeffs = methods.hantzko_et_al.calcPauliVector(matrix)

    maxDif = 1E-10
    for c1, c2, c3, c4 in zip(jone_coeffs, rome_coeffs, hama_coeffs, hant_coeffs):
        assert abs(c1-c2) < maxDif, "c1 c2 disagreed"
        assert abs(c1-c3) < maxDif, "c1 c3 disagreed"
        assert abs(c1-c4) < maxDif, "c1 c4 disagreed"

print("All tests passed\n")



'''
    Informal benchmarking
'''

print('Informal full-decomposition benchmarking (not logged)...')

for numQubits in range(1, 5+1):
    print('numQubits =', numQubits)

    for name, method in [
        ('jones',     methods.jones.calcPauliVector),
        ('romero',    methods.romero_et_al.calcPauliVector),
        ('hamaguchi', methods.hamaguchi_et_al.calcPauliVector),
        ('hantzko',   methods.hantzko_et_al.calcPauliVector)
    ]:
        durs = repeatedlyTime(method, numQubits, 10)
        av = sum(durs)/10.
        print(f'  {name}:\t{av} ns')



'''
    Meta log config
'''

if len(sys.argv[1:]) != 1:
    print('\nTo perform logged benchmarks, please supply the log output directory as a command-line arg.')
    quit()

dir = sys.argv[1]
if dir[-1] != '/':
    dir += '/'



'''
    Logged inner-product benchmarking between Jones and Romero
'''

print('\nBenchmarking inner-product...')
print('TODO')

# not sure this yet needs dedicated code - the full-decomp tests
# below inadvertently compare our inner-product times

outFn = dir + 'inner_prod.txt'



'''
    Logged full-decomposition benchmarking (all methods)
'''

# config
numReps = 100
minQubits = 1
maxQubits = 7

# output data
outFn = dir + 'full_decomp_all.txt'
results = {
    'numReps':numReps, 
    'minQubits': minQubits,
    'maxQubits': maxQubits,
    'reachedQubits': 0,
    'durations': {}
}

print(f"\nBenchmarking all methods at full-decomposition with {numReps} repetitions from {minQubits} to {maxQubits} qubits...")

for numQubits in range(minQubits, maxQubits+1):
    print(f'numQubits = {numQubits}')

    results['durations'][numQubits] = {}

    for name, method in [
        ('jones',     methods.jones.calcPauliVector),
        ('romero',    methods.romero_et_al.calcPauliVector),
        ('hamaguchi', methods.hamaguchi_et_al.calcPauliVector),
        ('hantzko',   methods.hantzko_et_al.calcPauliVector)
    ]:
        results['durations'][numQubits][name] = repeatedlyTime(method, numQubits, numReps)

    # save all results so far to file (overwriting, so job can be safely interrupted)
    results['reachedQubits'] = numQubits
    save_as_mma(results, outFn)



'''
    Further full-decomposition benchmarking of Hamaguchi and Hantzko
'''

# config
numReps = 100
minQubits = maxQubits + 1 # continue from previous tests
maxQubits = 15

# output data
outFn = dir + 'full_decomp_hamaguchi_hantzko.txt'
results = {
    'numReps':numReps, 
    'minQubits': minQubits,
    'maxQubits': maxQubits,
    'reachedQubits': 0,
    'durations': {}
}

print(f"\nFurther full-decomposition benchmarking of Ha* methods with {numReps} repetitions from {minQubits} to {maxQubits} qubits...")

for numQubits in range(minQubits, maxQubits+1):
    print(f'numQubits = {numQubits}')

    results['durations'][numQubits] = {}

    for name, method in [
        ('hamaguchi', methods.hamaguchi_et_al.calcPauliVector),
        ('hantzko',   methods.hantzko_et_al.calcPauliVector)
    ]:
        results['durations'][numQubits][name] = repeatedlyTime(method, numQubits, numReps)

    # save all results so far to file (overwriting, so job can be safely interrupted)
    results['reachedQubits'] = numQubits
    save_as_mma(results, outFn)





# TODO:
    # re-enable njit so we don't have to wait so long for benchmarking results lmao