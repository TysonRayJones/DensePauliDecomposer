'''
    Comparative runtime benchmarking of all methods 
    (with JIT) computing a small number of inner-products.
'''

import sys; sys.path.insert(0, '.')
from numba import config

from utils.benchmarker import benchmarkInnerProds
import methods.jones
import methods.new_romero
import methods.hamaguchi
import methods.hantzko


assert not bool(config.DISABLE_JIT), 'JIT must be enabled'
assert len(sys.argv[1:]) == 1, 'Must specify filename as command-line arg.'


methods = [
    ('Jones',     methods.jones.      calcInnerProds),
    ('Romero',    methods.new_romero. calcInnerProds),
    ('Hamaguchi', methods.hamaguchi.  calcInnerProds),
    ('Hantzko',   methods.hantzko.    calcInnerProds)
]

# maximum number of Pauli products to inner-product each matrix against,
# as a function of the number of qubits n
f=lambda n: 2**n

benchmarkInnerProds(
    sys.argv[1], methods, 
    minQubits=1, maxQubits=15, numReps=10, maxNumProdsFunc=f)