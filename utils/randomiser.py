
from numpy import array
from random import sample, choice, random


def getRandomMatrix(numQubits):
    dim = 1 << numQubits
    matrix = [[complex(random(), random()) for _ in range(dim)] for _ in range(dim)]

    matrix = array([array(r) for r in matrix])
    return matrix


def _getRandomPauliProd(numQubits, numNonIdentity):

    paulis = [0]*numQubits
    for i in sample(range(0,numQubits), numNonIdentity):
        paulis[i] = choice([1,2,3])

    return paulis


def getRandomPauliProdInd(numQubits, numNonIdentity):

    paulis = _getRandomPauliProd(numQubits, numNonIdentity)
    ind = sum(p * 4**n for n,p in enumerate(reversed(paulis)))
    return ind