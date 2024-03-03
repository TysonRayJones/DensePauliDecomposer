'''
An abridged implementation of Hamaguchi et al's algorithm presented in Lemma 4 Appendix B.1 here:
    https://arxiv.org/abs/2311.01362

shortened by Tyson Jones from its original source here:
    https://github.com/quantum-programming/RoM-handbook/blob/main/exputils/state/state_in_pauli_basis.py

by removing code irrelevant to our testing. Specifically, we:
    - additionally return imaginary component of Pauli vector
    - renormalise Pauli vector element outputs (dividing by 2^numQubits)
    - removed pre- and post-condition checking
    - removed python type clues (for consistency with other JIT'd methods)
    - renamed density_matrix to matrix (method confirmed compatible with arbitrary complex matrices)
    - removed function '_make_n_paulis()' and related imports not used by main algorithm
    - removed docstring
    - removed 'state_in_pauli_basis()' interface function
    - renamed '_state_in_pauli_basis_inplace_calculation' to '_calcAllCoeffs'
    - added 'calcPauliVector()' wrapper

    author:   hari64boli64
    uploaded: 27th Sep 2023

    editor: Tyson Jones
            tyson.jones.input@gmail.com
    edited: 2nd March 2024
'''


from numba import njit
import numpy as np



'''
    Bitwise functions (to enable compiling calcPauliCoeffs())
'''

@njit(inline='always')
def getLog2(n):
    r  = (n & 0xAAAAAAAA)  != 0
    r |= ((n & 0xFFFF0000) != 0) << 4
    r |= ((n & 0xFF00FF00) != 0) << 3
    r |= ((n & 0xF0F0F0F0) != 0) << 2
    r |= ((n & 0xCCCCCCCC) != 0) << 1
    return r



'''
    Decomposition
'''

@njit(cache=True)
def calcPauliVector(matrix):

    # must not mutate original matrix
    matrix = matrix.copy()

    size = len(matrix)
    n_qubit = getLog2(size)
    
    # The following code is based on FWHT like method.
    # Reference: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
    for k in range(n_qubit):
        shift = 1 << k
        for i_offset in range(0, size, shift * 2):
            for j_offset in range(0, size, shift * 2):
                for i in range(i_offset, i_offset + shift):
                    for j in range(j_offset, j_offset + shift):
                        I = matrix[i][j] + matrix[i + shift][j + shift]
                        Z = matrix[i][j] - matrix[i + shift][j + shift]
                        matrix[i][j] = I
                        matrix[i + shift][j + shift] = Z
                        X = matrix[i][j + shift] + matrix[i + shift][j]
                        Y = (matrix[i][j + shift] - matrix[i + shift][j]) * 1j
                        matrix[i][j + shift] = X
                        matrix[i + shift][j] = Y

    # The following code is based on z order curve.
    # Reference: https://en.wikipedia.org/wiki/Z-order_curve
    interlace_zeros = np.zeros(
        size, dtype=np.int64
    )  # interlace_zeros[0b1011] = 0b1000101
    for i in range(size):
        for s in range(n_qubit):
            if i & (1 << s):
                interlace_zeros[i] |= 1 << (2 * s)

    ret = np.zeros(size * size, np.complex128)
    norm = 1/float(size)
    for i in range(size):
        for j in range(size):
            index = (interlace_zeros[i] << 1) | interlace_zeros[j]
            ret[index] = matrix[i][j] * norm

    return ret



'''
    Inner-products
'''

@njit(cache=True)
def calcInnerProds(matrix, inds):
    all_coeffs = calcPauliVector(matrix)
    return [all_coeffs[i] for i in inds]
