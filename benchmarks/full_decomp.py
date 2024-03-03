'''
    Comparative runtime benchmarking of all methods 
    performing full-decomposition with JIT.
'''

import sys; sys.path.insert(0, '.')
from numba import config

from utils.benchmarker import benchmarkFullDecomp
import methods.jones
import methods.new_romero
import methods.hamaguchi
import methods.hantzko


assert not bool(config.DISABLE_JIT), 'JIT must be enabled'
assert len(sys.argv[1:]) == 1, 'Must specify filename as command-line arg.'


methods = [
    ('Jones',     methods.jones.      calcPauliVector),
    ('Romero',    methods.new_romero. calcPauliVector),
    ('Hamaguchi', methods.hamaguchi.  calcPauliVector),
    ('Hantzko',   methods.hantzko.    calcPauliVector)
]

benchmarkFullDecomp(sys.argv[1], methods, minQubits=1, maxQubits=15, numReps=100)