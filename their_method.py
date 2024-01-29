
'''
An abridged implementation of Romero et al's 'PD' algorithm, presented here:
    https://link.springer.com/article/10.1007/s11128-023-04204-w

shortened from its original source here:
    https://github.com/sebastianvromero/PauliComposer/blob/main/pauli_decomposer.py

by removing code irrelevant to our testing. Specifically, we:
- removed code for diagonal and strictly real matrices (we test only dense, complex matrices)
- removed the spawning of process pools, since we test in serial settings only
  (both our codes are embarrassingly parallel across distinct coefficients)
- changed the output structure to a flat array instead of a dictionary (we don't need Pauli strings)

    author:   Sebastian V. Romero
              sebastianvidalromero@gmail.com
    uploaded: Dec 5th 2023

    editor: Tyson Jones
            tyson.jones.input@gmail.com
    edited: 29th Jan 2024
'''


import numpy as np
import itertools as it
import scipy.sparse as ss
from numbers import Number


BINARY = {'I': '0', 'X': '1', 'Y': '1', 'Z': '0'}
PAULI_LABELS = ['I', 'X', 'Y', 'Z']


class PauliComposer:

    def __init__(self, entry: str, weight: Number = None):
        n = len(entry)
        self.n = n
        self.dim = 1<<n
        self.entry = entry.upper()
        self.paulis = list(set(self.entry))
        mat_ent = {0: 1, 1: -1j, 2: -1, 3: 1j}
        self.ny = self.entry.count('Y') & 3
        init_ent = mat_ent[self.ny]
        if weight is not None:
            init_ent *= weight
        self.init_entry = init_ent
        self.iscomplex = np.iscomplex(init_ent)
        rev_entry = self.entry[::-1]
        rev_bin_entry = ''.join([BINARY[ent] for ent in rev_entry])
        col_val = int(''.join([BINARY[ent] for ent in self.entry]), 2)
        col = np.empty(self.dim, dtype=np.int32)
        col[0] = col_val

        if weight is not None:
            if self.iscomplex:
                ent = np.full(self.dim, self.init_entry)
            else:
                ent = np.full(self.dim, float(self.init_entry))
        else:
            if self.iscomplex:
                ent = np.full(self.dim, self.init_entry, dtype=np.complex64)
            else:
                ent = np.full(self.dim, self.init_entry, dtype=np.int8)

        for ind in range(n):
            p = 1<<int(ind) 
            p2 = p<<1
            disp = p if rev_bin_entry[ind] == '0' else -p
            col[p:p2] = col[0:p] + disp 
            if rev_entry[ind] in ['I', 'X']:
                ent[p:p2] = ent[0:p]
            else:
                ent[p:p2] = -ent[0:p]

        self.col = col
        self.mat = ent

    def to_sparse(self):
        self.row = np.arange(self.dim)
        return ss.csr_matrix((self.mat, (self.row, self.col)),
                             shape=(self.dim, self.dim))

    def to_matrix(self):
        return self.to_sparse().toarray()


class PauliDecomposer:

    def __init__(self, H: np.ndarray):
        row, col = H.shape[0], H.shape[1]
        n_row, n_col = np.log2(row), np.log2(col)
        n = int(np.ceil(max(n_row, n_col)))
        self.size = 1<<n
        self.H_real = H.real
        self.H_imag = H.imag
        self.rows = np.arange(1<<n)
        self.coefficients = [0]*(1 << (2*n))

        combs = it.product(PAULI_LABELS, repeat=n)
        for i, comb in enumerate(combs):
            self.coefficients[i] = self.compute_weights_general(comb)

    def compute_weights_general(self, comb) -> None:
        value = 0
        entry = ''.join(comb)
        pauli_comp = PauliComposer(entry)
        cols, ent = pauli_comp.col, pauli_comp.mat
        for r in self.rows:
            coef, c = ent[r], cols[r]
            ham_term = self.H_real[c, r] + 1j*self.H_imag[c, r]
            if coef == 1:
                value += ham_term
            elif coef == -1:
                value -= ham_term
            else:
                value += coef * ham_term
        if not np.iscomplex(value):
            value = float(value)
        return value / self.size



'''
    Benchmarked code
'''

def get_all_coefficients(matrix):
    dec = PauliDecomposer(matrix)
    return dec.coefficients