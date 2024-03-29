# MNx: A Numeric Methods Library

`MNx` is a robust and efficient Python library designed to solve complex numerical problems. It provides a wide range of methods for solving linear systems, including the Gauss and Gauss-Jordan methods. The library is designed with flexibility in mind, allowing for the input of single arrays or lists of arrays.

## Features

- **Gauss Method**: Solve linear systems using the Gauss method.
- **Gauss-Jordan Method**: Solve linear systems using the Gauss-Jordan method.
- **Optimization**: Choose the best method for solving a system based on the given input.
- **Flexibility**: Accepts both single arrays and lists of arrays as input.

## Installation

You can install `mnx` using pip:

```bash
pip install mnx
```

## Usage

Here's a simple example of how to use `mnx`:

```python
import numpy as np
from mnx.exact_linear_methods.Solver import LinearSolver

# Define the system
A = np.array([[3, 2], [1, 2]])
b = np.array([2, 1])

# Create a GaussMethod instance
mnx = LinearSolver(A, b)

# Solve the system
print(mnx)
```

## License

`mnx` is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or feedback, please feel free to [contact us](mailto:juancamilogallego70@icloud.com).