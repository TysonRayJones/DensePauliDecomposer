'''
    Comparative runtime benchmarking of the Hamaguchi
    and Hantzko methods performing full-decomposition,
    beyond the practical regime for the Jones and Romero
    methods. This should be run with AND without jit 
    (disabled by export NUMBA_DISABLE_JIT=1) in case we
    are disadvantaging Hantzko with sub-optimal compilation
    (due to its recursive nature).
'''

import sys; sys.path.insert(0, '.')

from utils.benchmarker import benchmarkFullDecomp
import methods.hamaguchi
import methods.hantzko

assert len(sys.argv[1:]) == 1, 'Must specify filename as command-line arg.'


methods = [
    ('Hamaguchi', methods.hamaguchi.calcPauliVector),
    ('Hantzko', methods.hantzko.calcPauliVector)
]

benchmarkFullDecomp(sys.argv[1], methods, minQubits=1, maxQubits=15, numReps=100)