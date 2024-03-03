'''
An abridged implementation of Hantzko et al's 'TPD' algorithm, presented here:
    https://arxiv.org/abs/2310.13421

shortened by Tyson Jones from its original source here:
    https://github.com/HANTLUK/PauliDecomposition/blob/master/TensorizedPauliDecomposition.py

by removing code irrelevant to our testing. Specifically, we:
    - added just-in-time compilation via numba
    - removed power-of-2 padding (we always supply 2^N matrices)
    - removed superfluous creation of strings (e.g. 'XYZ') 
    - removed try/catch of matrix indexing (we always give numpy matrices)
    - removed sparse and 'nonzero' checks (confirmed never occurs in our tests)
    - make dim=1 basecase return immediately for clarity
    - removed docstring
    - replaced 'tab' indenting with '4 spaces'
    - removed superfluous 'coefficients' dict (replaced with tuple enumeration)
    - minor reformatting of algebra
    - renamed 'PauliDecomposition' to 'calcPauliVector()'

    author:   Lukas Hantzko
              lukas.hantzko@stud.uni-hannover.de
    uploaded: 11th Sep 2023

    editor: Tyson Jones
            tyson.jones.input@gmail.com
    edited: 2nd March 2024
'''


from numba import njit
import numpy as np
import math



'''
    Decomposition
'''


@njit(cache=True)
def calcPauliVector(matrix):

    matDim = matrix.shape[0]
    qBitDim = math.ceil(np.log(matDim)/np.log(2))

    # Output for dimension 1
    if qBitDim == 0:
        return [matrix[0,0]]

    # Calculates the tensor product coefficients via the sliced submatrices.
    halfDim = int(2**(qBitDim-1))
    coeff1 =  0.5  * (matrix[0:halfDim, 0:halfDim] + matrix[halfDim:,  halfDim:])
    coeffX =  0.5  * (matrix[halfDim:,  0:halfDim] + matrix[0:halfDim, halfDim:])
    coeffY = -0.5j * (matrix[halfDim:,  0:halfDim] - matrix[0:halfDim, halfDim:])
    coeffZ =  0.5  * (matrix[0:halfDim, 0:halfDim] - matrix[halfDim:,  halfDim:])

    # free matrix
    matrix = None

    # Recursion for the Submatrices
    coeffs = []
    for mat in (coeff1, coeffX, coeffY, coeffZ):
        subDec = calcPauliVector(mat)
        coeffs.extend(subDec)

    return coeffs



'''
    Inner-products
'''

@njit(cache=True)
def calcInnerProds(matrix, inds):
    all_coeffs = calcPauliVector(matrix.copy())
    return [all_coeffs[i] for i in inds]
