from time import process_time_ns
from numba import config

from .mmaformatter import save_as_mma
from .randomiser import getRandomMatrix, getRandomPauliProdInd


def repeatedlyTime(func, argFunc, numReps):
    durs = []

    for _ in range(numReps):
        args = argFunc()

        t1 = process_time_ns()
        func(*args)
        t2 = process_time_ns()

        durs.append(t2 - t1)
    
    return durs



def benchmarkFullDecomp(fn, namedMethods, minQubits, maxQubits, numReps=100):
    
    results = {
        'numReps':numReps, 
        'minQubits': minQubits,
        'maxQubits': maxQubits,
        'reachedQubits': -1,
        'durations': {},
        'jit': not bool(config.DISABLE_JIT)
    }

    for numQubits in range(minQubits, maxQubits+1):
        print(f'numQubits = {numQubits}')

        results['durations'][numQubits] = {}

        for name, method in namedMethods:
            argFunc = lambda n=numQubits : (getRandomMatrix(n),)
            results['durations'][numQubits][name] = repeatedlyTime(method, argFunc, numReps)

        # save all results so far to file (overwriting, so job can be safely interrupted)
        results['reachedQubits'] = numQubits
        save_as_mma(results, fn)



def benchmarkInnerProds(fn, namedMethods, minQubits, maxQubits, maxNumProdsFunc, numReps=10):

    indent = '  '

    results = {
        'numReps': numReps, 
        'minQubits': minQubits,
        'maxQubits': maxQubits,
        'reachedQubits': -1,
        'durations': {},
        'structure': 'durs -> numQubits -> numNondId (1 to numQubits) -> numProds (1 to func(numQubits) -> method name -> list of dirs',
        'jit': not bool(config.DISABLE_JIT)
    }

    for numQubits in range(minQubits, maxQubits+1):
        print('numQubits =', numQubits)
        results['durations'][numQubits] = {}

        for numNonId in range(1, numQubits+1):
            print(f'{indent}numNonId = {numNonId}')
            results['durations'][numQubits][numNonId] = {}

            for numProds in range(1, maxNumProdsFunc(numQubits)+1):
                print(f'{indent*2}numProds = {numProds}')
                results['durations'][numQubits][numNonId][numProds] = {}

                for name, method in namedMethods:
                    reps = 10
                    args = lambda n=numQubits,m=numNonId,p=numProds : (
                        getRandomMatrix(n),
                        [getRandomPauliProdInd(n,m) for _ in range(p+1)] )
                    
                    durs = repeatedlyTime(method, args, reps)
                    results['durations'][numQubits][numNonId][numProds][name] = durs

                    # log progress
                    results['reachedProds'] = numProds
                    results['reachedNonId'] = numNonId
                    results['reachedQubits'] = numQubits
                    
                    # save all results so far to file (overwriting, so job can be safely interrupted)
                    save_as_mma(results, fn)