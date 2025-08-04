# YantraShiksha 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-numpy%20matplotlib-green.svg)](https://pypi.org/project/numpy/)

> **A modern deep learning library designed for flexibility, extensibility, and educational clarity.**

YantraShiksha provides a full stack for building, training, and experimenting with neural networks, including both core tensor operations and high-level model APIs. The library is implemented in Python with performance-critical components in C++ (via pybind11), and features a blend of Sanskrit-inspired naming and modern deep learning concepts.

## ✨ Features

- 🚀 **Autograd Engine**: Automatic differentiation for tensor operations, supporting gradients for all core mathematical operations and custom layers
- 📊 **Tensor Library (`Tanitra`)**: Numpy-based tensor class with support for broadcasting, slicing, matrix multiplication, and a variety of activation and utility functions
- 🏗️ **Layer API (`Parata`)**: Modular layer classes including dense, input, output, normalization, convolutional, pooling, LSTM, and transformer components
- 🤖 **Model API (`Pratirup`)**: Sequential and word embedding models, with easy-to-use training loops and support for custom optimizers
- 🔄 **Transformer Support**: Self-attention, multi-headed attention, and positional encoding layers for building transformer architectures
- 📚 **Educational Focus**: Clear, readable code with a focus on learning and experimentation
- ⚡ **High-Performance C++ Backend**: Ganit tensor core for optimized operations

---

## 🚀 Getting Started

### Installation
```
**Requirements:** Python 3.8+, numpy, matplotlib, pybind11 (only for building from source)

#### Method 1: Direct File Inclusion (Recommended)
Copy these files to your project:
- `Tanitra.py` - Python tensor implementation
- `Parata.py` - Layer definitions  
- `Pratirup.py` - Model APIs
- `Ganit.cp313-win_amd64.pyd` - Pre-compiled C++ tensor core

#### Method 2: Build Custom Binary
For different platforms/Python versions:
```bash
# Copy Ganit/ directory to your project
pip install pybind11 numpy
python setup.py build_ext --inplace
# Copy generated .pyd/.so file to your project
```

#### Method 3: Development Setup
For contributing or modifications:
```bash
git clone <repository-url>
cd YantraShiksha
pip install numpy matplotlib pybind11
python setup.py build_ext --inplace
```

### Quick Example

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

**Usage:**
```python
import Tanitra
import Parata
import Pratirup
import Ganit  # C++ tensor core
```

---

## 📚 Documentation

### Core Components

#### 🧮 Tensors & Autograd

- `Tanitra`: Core tensor class with autograd support
- Supports: `+`, `-`, `*`, `/`, `@` (matmul), slicing, flatten, transpose, and more
- Activation functions: `sigmoid`, `relu`, `tanh`, `softmax`, etc.
- Utility functions: `mean`, `square`, `convolution2d`, `pooling2d`, etc.

#### ⚡ Ganit: High-Performance C++ Tensor Core

The tensor operations are being migrated to a high-performance C++ implementation exposed via pybind11 bindings. This provides:

- **Performance**: Optimized C++ backend for tensor operations
- **Memory Efficiency**: Better memory management for large tensors
- **Extensibility**: Easy to add new operations in C++
- **API Consistency**: Same Python interface with improved performance

##### Current Ganit API (Work in Progress)

```python
import Ganit

# Create tensors
a = Ganit.Tanitra([1, 2, 3, 4])
b = Ganit.Tanitra([[1, 2], [3, 4]])

# Basic operations
c = a + b
d = a * b
e = a @ b  # matrix multiplication

# Autograd
e.backward()
grad = e.grad()

# Trigonometric functions
f = Ganit.sin(a)
g = Ganit.cos(a)
h = Ganit.tan(a)

# Utility functions
reshaped = a.reshape([2, 2])
transposed = a.T()
conv_result = Ganit.convolution(a, b, stride=1)
```

> **Note**: The Ganit API is actively being developed and will have improved functionality and performance in future releases.

#### 🏗️ Layers

- `PraveshParata`: Input layer
- `GuptaParata`: Dense (fully connected) layer with activations
- `NirgamParata`: Output layer
- `Samasuchaka`: Normalization (z-score, min-max)
- `ConvLayer2D`: 2D convolutional layer
- `MaxPoolingLayer2D`: 2D max pooling
- `LSTM`: Long Short-Term Memory block
- `SelfAttention`, `MultiHeadedAttention`, `PositionalEncoding`: Transformer components

#### 🤖 Models

- `AnukramikPratirup`: Sequential model API
- `ShabdAyamahPratirup`: Word embedding model (CBOW, skip-gram)

---

## 🔧 Advanced Features

- **Custom Backpropagation**: Easily define new operations and their gradients
- **Extensible Layers**: Add your own layers by subclassing `Parata`
- **Transformer Support**: Build transformer models with attention and positional encoding
- **High-Performance C++ Backend**: Ganit tensor core for optimized operations



---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Inspired by PyTorch, TensorFlow, and the spirit of open-source learning
- Sanskrit-inspired naming for educational and cultural flavor

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
Made with ❤️ for the deep learning community
</div>
