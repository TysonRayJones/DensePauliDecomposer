
from .randomiser import getRandomMatrix


_MAX_DIF = 1E-10


def testMethods(methods, maxNumQubits=5):

    print('Testing correctnesss...')

    for numQubits in range(1, maxNumQubits+1):
        print('numQubits =', numQubits)

        matrix = getRandomMatrix(numQubits)
        results = [f(matrix) for f in methods]

        for coeffs in zip(*results):
            for c in coeffs[1:]:
                assert abs(c-coeffs[0]) < _MAX_DIF, "Test failed"
    
    print("All tests passed\n")