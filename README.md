# YantraShiksha

**YantraShiksha** is a modern deep learning library designed for flexibility, extensibility, and educational clarity. It provides a full stack for building, training, and experimenting with neural networks, including both core tensor operations and high-level model APIs. The library is implemented in Python with performance-critical components in C++ (via pybind11), and features a blend of Sanskrit-inspired naming and modern deep learning concepts.

---

## Features

- **Autograd Engine**: Automatic differentiation for tensor operations, supporting gradients for all core mathematical operations and custom layers.
- **Tensor Library (`Tanitra`)**: Numpy-based tensor class with support for broadcasting, slicing, matrix multiplication, and a variety of activation and utility functions.
- **Layer API (`Parata`)**: Modular layer classes including dense, input, output, normalization, convolutional, pooling, LSTM, and transformer components.
- **Model API (`Pratirup`)**: Sequential and word embedding models, with easy-to-use training loops and support for custom optimizers.
- **Transformer Support**: Self-attention, multi-headed attention, and positional encoding layers for building transformer architectures.
- **Educational Focus**: Clear, readable code with a focus on learning and experimentation.

---

## Installation

### Option 1: Use Pre-compiled Binary (Recommended)
The project includes a pre-compiled `Ganit.cp313-win_amd64.pyd` file for Windows with Python 3.13. Simply install the Python dependencies:

```bash
pip install numpy matplotlib
```

### Option 2: Build from Source
If you need to build the C++ extensions for a different platform or Python version:

```bash
pip install numpy matplotlib pybind11
python setup.py build_ext --inplace
```

Requirements:
- Python 3.8+
- numpy
- matplotlib
- pybind11 (only for building from source)

---

## Quick Example

```python
import Tanitra
import Parata
import Pratirup

# Build a simple feedforward network
model = Pratirup.AnukramikPratirup([
    Parata.PraveshParata((2,)),
    Parata.GuptaParata(8, 'relu'),
    Parata.NirgamParata(1, 'sigmoid')
])

# Dummy data
import numpy as np
X = Tanitra.Tanitra(np.random.randn(100, 2))
y = Tanitra.Tanitra(np.random.randint(0, 2, (100, 1)))

# Train
model.learn(X, y, epochs=100, lr=0.01)
```

---

## Core Components

### Tensors & Autograd

- `Tanitra`: Core tensor class with autograd support.
- Supports: `+`, `-`, `*`, `/`, `@` (matmul), slicing, flatten, transpose, and more.
- Activation functions: `sigmoid`, `relu`, `tanh`, `softmax`, etc.
- Utility functions: `mean`, `square`, `convolution2d`, `pooling2d`, etc.

### Layers

- `PraveshParata`: Input layer
- `GuptaParata`: Dense (fully connected) layer with activations
- `NirgamParata`: Output layer
- `Samasuchaka`: Normalization (z-score, min-max)
- `ConvLayer2D`: 2D convolutional layer
- `MaxPoolingLayer2D`: 2D max pooling
- `LSTM`: Long Short-Term Memory block
- `SelfAttention`, `MultiHeadedAttention`, `PositionalEncoding`: Transformer components

### Models

- `AnukramikPratirup`: Sequential model API
- `ShabdAyamahPratirup`: Word embedding model (CBOW, skip-gram)

---

## Advanced Features

- **Custom Backpropagation**: Easily define new operations and their gradients.
- **Extensible Layers**: Add your own layers by subclassing `Parata`.
- **Transformer Support**: Build transformer models with attention and positional encoding.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Inspired by PyTorch, TensorFlow, and the spirit of open-source learning.
- Sanskrit-inspired naming for educational and cultural flavor.

---

## Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

---
