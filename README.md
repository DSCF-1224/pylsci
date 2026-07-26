# pylsci

[![CI](https://github.com/DSCF-1224/pylsci/actions/workflows/ci.yml/badge.svg)](https://github.com/DSCF-1224/pylsci/actions/workflows/ci.yml)

Python implementation of Least Squares Reference Circle (LSCI) fitting and roundness evaluation

## Installation

`pylsci` is not published to PyPI. Install directly from this repository.

### pip

```bash
pip install "git+https://github.com/DSCF-1224/pylsci.git"
```

### uv

```bash
uv add "git+https://github.com/DSCF-1224/pylsci.git"
```


## Usage

Inputs must be 1-dimensional.
See [Algorithm documentation](docs/algorithm.md#current-limitations) for details.

### `NumPy` backend

```python
import numpy as np

from pylsci.numpy_backend import fit

theta = 2 * np.pi * np.arange(360) / 360

result = fit(
    x=np.cos(theta),
    y=np.sin(theta),
)

print(result.center.x)
print(result.center.y)
print(result.radius)
print(result.roundness)
```

### `PyTensor` backend

```python
from pytensor import tensor as pt

from pylsci.pytensor_backend import fit

theta = 2 * np.pi * pt.arange(360) / 360

result = fit(
    x=pt.cos(theta),
    y=pt.sin(theta),
)

print(result.center.x.eval())
print(result.center.y.eval())
print(result.radius.eval())
print(result.roundness.eval())
```

`center`, `radius`, and `roundness` are symbolic ([`TensorVariable`](https://pytensor.readthedocs.io/en/stable/library/tensor/basic.html#pytensor.tensor.TensorVariable));
call [`.eval()`](https://pytensor.readthedocs.io/en/stable/library/graph/graph.html#pytensor.graph.basic.Variable.eval) to obtain numeric values, or use [`pytensor.function`](https://pytensor.readthedocs.io/en/stable/library/compile/function.html#pytensor.compile.maker.function) to compile
a reusable function.

## Algorithm

See [Algorithm documentation](docs/algorithm.md)
for the mathematical derivation and implementation details.
