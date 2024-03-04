'''
    Comparative runtime benchmarking of the original*
    and refactored Romero et al PD implementations.
    This should be run with and without jit 
    (disabled by export NUMBA_DISABLE_JIT=1) even though
    the original* implementation is not Numba-compatible,
    so that the pure-python _and_ compiled speedups 
    are known.

    *(well still simplified from the Github upload)
'''

import sys; sys.path.insert(0, '.')

from utils.benchmarker import benchmarkFullDecomp

import methods.old_romero
import methods.new_romero


assert len(sys.argv[1:]) == 1, 'Must specify filename as command-line arg.'


methods = [
    ('New Romero', methods.new_romero.calcPauliVector),
    ('Old Romero', methods.old_romero.calcPauliVector)
]

benchmarkFullDecomp(sys.argv[1], methods, minQubits=1, maxQubits=15, numReps=100)