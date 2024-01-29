'''
    A Python3 implementation of our algorithm (calcPauliCoeffFast),
    including an optimised implementation with subroutines unpacked 
    (calcPauliCoeffFaster), and a simpler but slower algorithm 
    for comparison (calcPauliCoeffSlow).

    author: Tyson Jones
            tyson.jones.input@gmail.com
    date:   29th Jan 2024
'''


'''
    Bitwise functions
'''

def getPowerOf2(n):
    return 1 << n

def getLog2(n):
    r  = (n & 0xAAAAAAAA)  != 0
    r |= ((n & 0xFFFF0000) != 0) << 4
    r |= ((n & 0xFF00FF00) != 0) << 3
    r |= ((n & 0xF0F0F0F0) != 0) << 2
    r |= ((n & 0xCCCCCCCC) != 0) << 1
    return r

def getBit(n, t):
    return (n >> t) & 1

def getGrayCode(n):
    return n ^ (n >> 1)

def getChangedBit(i, j):
    return getLog2(i ^ j)


'''
    Pauli functions
'''

PAULI_MATRICES = [
    [[1,0],[0,1]],
    [[0,1],[1,0]],
    [[0,-1j],[1j,0]],
    [[1,0],[0,-1]]
]

def getPauliFlag(n, t):
    b0 = getBit(n, 2*t)
    b1 = getBit(n, 2*t+1)
    return (b1 << 1) | b0

def getMaskOfXY(n, numQubits):
    mask = 0

    for t in range(numQubits):
        p = getPauliFlag(n, t)
        b = 0 < p < 3
        mask |= b << t

    return mask

def getPauliElem(n, t, i, mask):
    p = getPauliFlag(n, t)
    a = getBit(i, t)
    b = a ^ getBit(mask, t)
    e = PAULI_MATRICES[p][a][b]
    return e

def getFactor(n, i, mask, numQubits):
    factor = 1

    for t in range(numQubits):
        factor *= getPauliElem(n, t, i, mask)

    return factor



'''
    Algorithms
'''

def calcPauliCoeffSlow(n, matrix, numQubits):

    coeff = 0
    mask = getMaskOfXY(n, numQubits)
    dim = getPowerOf2(numQubits)

    # evaluate each factor_i independently
    for i in range(dim):
        factor = getFactor(n, i, mask, numQubits)
        coeff += factor * matrix[i ^ mask][i]
    
    coeff /= float(dim)
    return coeff


def calcPauliCoeffFast(n, matrix, numQubits):

    coeff = 0
    mask = getMaskOfXY(n, numQubits)
    dim = getPowerOf2(numQubits)

    # evaluate factor_0 in full
    factor = getFactor(n, 0, mask, numQubits)
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


def calcPauliCoeffFaster(n, matrix, numQubits):

    # algorithmically equivalent to calcPauliCoeffFast()
    coeff = 0
    dim = 1 << numQubits

    # build mask
    mask = 0
    k = n
    for t in range(numQubits):
        mask |= ((k & 1) ^ ((k >> 1) & 1)) << t
        k >>= 2

    # evaluate factor_0 in full
    factor = 1
    k = n
    for t in range(numQubits):
        p = (k & 1) | (k & 2)
        factor *= PAULI_MATRICES[p][0][(mask >> t) & 1]
        k >>= 2

    coeff += factor * matrix[mask][0]

    # remember previous Gray code
    j = 0

    # recurrently evaluate factor_i
    for k in range(1, dim):

        # get current and previous Gray codes
        i = k ^ (k >> 1)

        # find bit t where codes differ
        r = i ^ j
        t  = (r & 0xAAAAAAAA)  != 0
        t |= ((r & 0xFFFF0000) != 0) << 4
        t |= ((r & 0xFF00FF00) != 0) << 3
        t |= ((r & 0xF0F0F0F0) != 0) << 2
        t |= ((r & 0xCCCCCCCC) != 0) << 1

        # find Pauli at t
        s = n >> (2*t)
        p = (s & 2) | (s & 1)

        # find new factor term
        a = (i >> t) & 1
        b = a ^ ((mask >> t) & 1)
        c = PAULI_MATRICES[p][a][b]

        # find old factor term
        a = (j >> t) & 1
        b = a ^ ((mask >> t) & 1)
        d = PAULI_MATRICES[p][a][b]

        # evaluate factor_i
        factor *= c / d
        coeff += factor * matrix[i ^ mask][i]

        # remember previous Gray code
        j = i

    coeff /= float(dim)
    return coeff



'''
    Benchmarked code
'''

def get_all_coefficients(matrix):

    numQubits = getLog2(len(matrix))
    numCoeffs = getPowerOf2(2*numQubits)

    coeffs = [0] * numCoeffs
    for n in range(numCoeffs):
        coeffs[n] = calcPauliCoeffFaster(n, matrix, numQubits)

    return coeffs