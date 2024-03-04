'''
A revised implementation of Romero et al's 'PD' algorithm, presented here:
    https://link.springer.com/article/10.1007/s11128-023-04204-w

significantly refactored from its original object-oriented form here:
    https://github.com/sebastianvromero/PauliComposer/blob/main/pauli_decomposer.py

which achieves a speedup of ~4x on my 2017 Macbook.

Specifically, we:
    - removed superfluous use of objects ('PauliDecomposer', etc)
    - replaced superfluous string and dictionary manipulation with bitwise algebra
    - removed superfluous memory structures ('H_real', etc)
    - removed code for diagonal and strictly real matrices (we test only dense, complex matrices)
    - removed the spawning of process pools, since we test in serial settings only
    - added just-in-time compilation with Numba 
    - deferred evaluation of specific coefficients
    - added calcPauliVector() wrapper

    author:   Sebastian V. Romero
              sebastianvidalromero@gmail.com
    uploaded: Dec 5th 2023

    editor: Tyson Jones
            tyson.jones.input@gmail.com
    edited: 3rd March 2024
'''


from numba import njit
import numpy as np



'''
    Helper functions
'''

@njit(inline='always')
def getPowerOf2(n):
    return 1 << n

@njit(inline='always')
def getBit(n, t):
    return (n >> t) & 1

@njit(inline='always')
def getLog2(n):
    r  = (n & 0xAAAAAAAA)  != 0
    r |= ((n & 0xFFFF0000) != 0) << 4
    r |= ((n & 0xFF00FF00) != 0) << 3
    r |= ((n & 0xF0F0F0F0) != 0) << 2
    r |= ((n & 0xCCCCCCCC) != 0) << 1
    return r

@njit(inline='always')
def getPauliFlag(n, t):
    b0 = getBit(n, 2*t)
    b1 = getBit(n, 2*t+1)
    return (b1 << 1) | b0

@njit(inline='always')
def getNumFromBits(bits):
    # numba can't handle sum() comprehension :'c
    num = 0
    for i, b in enumerate(bits):
        num += b * getPowerOf2(i)
    return num



'''
    Romero et al PD simplified
'''

@njit(cache=True)
def _getIndArray(paulis):

    # allocate 2^N memory for 'colInds'
    dim = getPowerOf2(len(paulis))
    colInds = np.empty(dim, dtype=np.int32)

    # set first col elem to integer with bits paulis[n]==(X|Y)
    colInds[0] = getNumFromBits([paulis[-i-1] in [1,2] for i in range(len(paulis))])
    
    # set remaining colInds
    for i in range(len(paulis)):
        j = getPowerOf2(i)
        s = 1 if paulis[-i-1] in [0,3] else -1 # sign = IZ vs XY
        colInds[j:j<<1] = colInds[0:j] + s * j

    return colInds


@njit(cache=True)
def _getCoeffArray(paulis):

    # initial 'coeffs' elem informed by Y=2 parity
    numY = paulis.count(2)
    elem = [1, -1j, -1, 1j][numY & 3] #  = %4

    # allocate and fill 'coeffs' 2^N memory with elem (avoid complex type if possible)
    #   elemType = np.complex64 if np.iscomplex(elem) else np.int8
    # actually we cannot avoid complex when compiling, since the type must be a priori known
    elemType = np.complex64
    dim = getPowerOf2(len(paulis))
    coeffs = np.full(dim, elem, dtype=elemType)

    # overwrite coeffs
    for i in range(len(paulis)):
        j = getPowerOf2(i)
        s = 1 if paulis[-i-1] in [0,1] else -1 # sign = IX vs YZ
        coeffs[j:j<<1] = s * coeffs[0:j]

    return coeffs


@njit(cache=True)
def _calcPauliCoeff(n, matrix):

    dim = len(matrix)
    numPaulis = getLog2(dim)
                        
    paulis = [getPauliFlag(n,t) for t in range(numPaulis-1,-1,-1)]
    colInds = _getIndArray(paulis)
    factors = _getCoeffArray(paulis)

    coeff = 0
    for r in range(len(matrix)):
        c, fac = colInds[r], factors[r]
        coeff += fac * matrix[c,r] / float(dim)

    return coeff



'''
    Inner product
'''

@njit(cache=True)
def calcInnerProds(matrix, inds):
    return [_calcPauliCoeff(n, matrix) for n in inds]



'''
    Decomposition
'''

@njit(cache=True)
def calcPauliVector(matrix):
    dim = len(matrix)**2
    vec = [_calcPauliCoeff(n, matrix) for n in range(dim)]
    return vec