'''
    Memory benchmarking of all methods, with and without JIT,
    computing a single inner-product
'''

import sys; sys.path.insert(0, '.')
from numba import config

from utils.benchmarker import memoryProfileInnerProds

import methods.jones
import methods.old_romero
import methods.new_romero
import methods.hamaguchi
import methods.hantzko


assert len(sys.argv[1:]) == 2, 'Must specify filename-prefix and method name as command-line args.'

methods = {
    'Jones':      methods.jones.      calcInnerProds,
    'Romero':     methods.new_romero. calcInnerProds,
    'Romero_old': methods.new_romero. calcInnerProds,
    'Hamaguchi':  methods.hamaguchi.  calcInnerProds,
    'Hantzko':    methods.hantzko.    calcInnerProds
}

name = sys.argv[2]
method = methods[name]
fn = f'{sys.argv[1]}_{name}_{"no_" if bool(config.DISABLE_JIT) else ""}jit.txt'

memoryProfileInnerProds(fn, name, method, minQubits=1, maxQubits=30, numReps=100)