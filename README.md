[Decomposing dense matrices into dense Pauli tensors](https://arxiv.org/abs/2401.16378)
==================================================

> Tyson Jones
> - Department of Materials, University of Oxford
> - Institute of Physics, EPFL
> - Quantum Motion Technologies Ltd

This repository contains a `Python` implementation of the algorithm presented by [this manuscript](https://arxiv.org/abs/2401.16378) for decomposing an arbitrary complex matrix into a weighted sum of Pauli strings.
Our main algorithm is defined as function `calcPauliCoeffFast()` in [`our_method.py`](our_method.py), with an optimised implementation defined as `calcPauliCoeffFaster()`. We include benchmarking in [`benchmarking.py`](benchmarking.py) against the [`PauliDecomposer`](https://link.springer.com/article/10.1007/s11128-023-04204-w) with code modified from its [source repo](https://github.com/sebastianvromero/PauliComposer).

If this code is useful to you, feel free to cite
```
@misc{jones2024decomposing,
      title={Decomposing dense matrices into dense Pauli tensors}, 
      author={Tyson Jones},
      year={2024},
      eprint={2401.16378},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```


# Usage

Using our algorithm requires a single import:
```Python
from our_method import calcPauliCoeffFaster
```

Given a complex matrix `G` with dimension `2^N`-by-`2^N`, instantiated as a list-of-lists or as a `numpy` `ndarray`, you can call
```Python
c = calcPauliCoeffFaster(n, G, N)
```
to compute the complex coefficient `c` of the `n`-th Pauli tensor.

To compute a `4^N`-length list of all coefficients, you can call
```Python
c_all = get_all_coefficients(G)
```
but beware that this loop has not been explicitly parallelised.

To perform the benchmarking, simply run
```bash
python3 benchmarking.py
```
which will log to file `results.txt` in a format which can be passed by `Mathematica`'s [`Get[]`](https://reference.wolfram.com/language/ref/Get.html) function. Benchmarking can be interrupted at any time, preserving the results so far. To change the benchmarked configuration, edit the variables at the bottom of [`benchmarking.py`](benchmarking.py).


# License

This repository is licensed under the terms of the MIT license.
