from numba import njit
from numpy import array



'''
    Bitwise functions
'''

#@njit(inline='always')
def getPowerOf2(n):
    return 1 << n

#@njit(inline='always')
def getLog2(n):
    r  = (n & 0xAAAAAAAA)  != 0
    r |= ((n & 0xFFFF0000) != 0) << 4
    r |= ((n & 0xFF00FF00) != 0) << 3
    r |= ((n & 0xF0F0F0F0) != 0) << 2
    r |= ((n & 0xCCCCCCCC) != 0) << 1
    return r

#@njit(inline='always')
def getBit(n, t):
    return (n >> t) & 1

#@njit(inline='always')
def getGrayCode(n):
    return n ^ (n >> 1)

#@njit(inline='always')
def getChangedBit(i, j):
    return getLog2(i ^ j)



'''
    Pauli functions
'''

PAULI_MATRICES = array([
    [[1,0],[0,1]],
    [[0,1],[1,0]],
    [[0,-1j],[1j,0]],
    [[1,0],[0,-1]]
])

#@njit(inline='always')
def getPauliFlag(n, t):
    b0 = getBit(n, 2*t)
    b1 = getBit(n, 2*t+1)
    return (b1 << 1) | b0

#@njit(inline='always')
def getMaskOfXY(n, numQubits):
    mask = 0

    for t in range(numQubits):
        p = getPauliFlag(n, t)
        b = 0 < p < 3
        mask |= b << t

    return mask

#@njit(inline='always')
def getPauliElem(n, t, i, mask):
    p = getPauliFlag(n, t)
    a = getBit(i, t)
    b = a ^ getBit(mask, t)
    e = PAULI_MATRICES[p][a][b]
    return e

#@njit(inline='always')
def getFactor(n, i, mask, numQubits):
    factor = 1

    for t in range(numQubits):
        factor *= getPauliElem(n, t, i, mask)

    return factor



'''
    Inner product
'''

#@njit(cache=True)
def calcPauliCoeff(n, matrix):
    '''
        Calculates the coefficient of the n-th 
        Pauli basis string of the matrix's
        Pauli decomposition (i.e. the inner-
        product of matrix with the n-th string)
        recurrently across the contributing
        elements of the original matrix. This can
        be made approximately twice as fast by
        in-lining the below bitwise functions.
    '''

    coeff = 0

    dim = len(matrix)
    logDim = getLog2(dim)
    mask = getMaskOfXY(n, logDim)

    # evaluate factor_0 in full
    factor = getFactor(n, 0, mask, logDim)
    coeff += factor * matrix[mask][0]

    # recurrently evaluate factor_i
    for k in range(1, dim):
        i = getGrayCode(k)
        j = getGrayCode(k - 1)
        t = getChangedBit(i, j)
        c = getPauliElem(n, t, i, mask)
        d = getPauliElem(n, t, j, mask)
        f = i ^ mask

        # by revising one term in factor's product
        factor *= c / d
        coeff += factor * matrix[f][i]

    coeff /= float(dim)
    return coeff



'''
    Decomposition
'''

##@njit(cache=True)
def calcPauliVector(matrix):
    dim = len(matrix)**2
    vec = [calcPauliCoeff(n, matrix) for n in range(dim)]
    return vec