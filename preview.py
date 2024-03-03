'''
To disable JIT, set env var:

    export NUMBA_DISABLE_JIT=1

'''

from numba import config
from time import sleep

from utils.correcter import testMethods
from utils.benchmarker import repeatedlyTime
from utils.randomiser import getRandomMatrix, getRandomPauliProdInd
from utils.memorymeasurer import getMemory

import methods.jones
import methods.old_romero
import methods.new_romero
import methods.hamaguchi
import methods.hantzko

indent = '  '



'''
    Env reporting
'''

print('Runtime environment:')
print('\tJIT:', not bool(config.DISABLE_JIT), '\n')
if bool(config.DISABLE_JIT):
    print(
        'Warning: no-compilation will disproportionately disadvantage the Jones ' +
        'method due to no inlining of the many function calls performed in an ' +
        'exponentially-large tight loop\n')

sleep(1)



'''
    Testing correctness
'''

testMethods([
    methods.jones.calcPauliVector, 
    methods.new_romero.calcPauliVector,
    methods.old_romero.calcPauliVector,
    methods.hamaguchi.calcPauliVector,
    methods.hantzko.calcPauliVector
])

sleep(1)



'''
    Full-decomp benchmarking
'''

print('Informal full-decomposition benchmarking (not logged)...')

for numQubits in range(1, 4+1):
    print('numQubits =', numQubits)

    for name, method in [
        ('romero (old)', methods.old_romero. calcPauliVector),
        ('romero',       methods.new_romero. calcPauliVector),
        ('jones',        methods.jones.      calcPauliVector),
        ('hantzko',      methods.hantzko.    calcPauliVector),
        ('hamaguchi',    methods.hamaguchi.  calcPauliVector)
    ]:
        reps = 10
        args = lambda n=numQubits : (getRandomMatrix(n),)
        durs = repeatedlyTime(method, args, reps)
        av = sum(durs)/float(reps)
        print(f'{indent}{name+":":12}{indent}{av:12} ns')

sleep(2)



'''
    Partial-decomp benchmarking
'''

print('\nInformal partial-decomposition benchmarking (not logged)...')

for numQubits in [5, 6, 7]:
    print('numQubits =', numQubits)

    for numNonId in (1, numQubits//2, numQubits):
        print(f'{indent}numNonId = {numNonId}')

        for numProds in [1, 10, numQubits**2, 2**numQubits]:
            print(f'{indent*2}numProds = {numProds}')

            for name, method in [
                ('romero',       methods.new_romero. calcInnerProds),
                ('jones',        methods.jones.      calcInnerProds),
                ('hantzko',      methods.hantzko.    calcInnerProds),
                ('hamaguchi',    methods.hamaguchi.  calcInnerProds)
            ]:
                reps = 10
                args = lambda n=numQubits,m=numNonId,p=numProds : (
                    getRandomMatrix(n),
                    [getRandomPauliProdInd(n,m) for _ in range(p+1)] )
    
                durs = repeatedlyTime(method, args, reps)
                av = sum(durs)/float(reps)
                print(f'{indent*3}{name+":":12}{indent}{av:12} ns')

sleep(1)



'''
    Test of memory querying
'''

print('\nmemory usage:')
print(getMemory())



'''
    Logged benchmarking
'''

print('\nTo run logged benchmarking, run each file in benchmarks/\n')