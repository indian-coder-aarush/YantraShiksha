import cupy as cp
import Tanitra
import math
import Ganit

# Demo for testing Ganit.Tanitra operations
a = Ganit.Tanitra([2,2], 3)
b = Ganit.Tanitra([2,2], 3)
c = Ganit.Tanitra([2,2], 2)

e = Ganit.__sub__(a, b)  # subtraction
f = Ganit.__add__(a, b)  # addition
g = Ganit.__add__(e, f)  # more operations
g.backward()             # backward pass to compute gradients
a.grad().print()         # print gradients of a

# ===== Dense Layer Base Class =====
class Parata:
    def __init__(self, input_shape=None):
        self.params = {}
        self.input_shape = input_shape
        self.output_shape = None

    def forward(self):
        raise NotImplementedError

# ===== Input Layer =====
class PraveshParata:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.params = {}

    def forward(self, data):
        if data.shape == self.input_shape:
            return data
        else:
            raise ValueError("Shape of input layer and the input data should be the same.")

# ===== Dense Hidden Layer (GuptaParata) =====
class GuptaParata:
    def __init__(self, n_neurons, activation, input_shape=None):
        if isinstance(n_neurons, tuple):
            raise ValueError("Only 1D output is allowed.")
        self.output_shape = (n_neurons,)
        self.params = {}
        self.n_neurons = n_neurons
        self.input_shape = None
        self.activation = activation
        self.input_output_learned = False

    def forward(self, input):
        # Initialize weights and biases on first call
        if not self.input_output_learned:
            self.input_shape = input.shape
            self.params = {
                "weights": Tanitra.Tanitra(cp.random.randn(math.prod(self.input_shape), self.n_neurons) *
                                           1 / cp.sqrt(math.prod(self.input_shape))),
                "biases": Tanitra.Tanitra(cp.random.randn(self.n_neurons) * 0.01)
            }
            self.input_output_learned = True

        input = input.flatten()

        # Check input shape matches
        if Tanitra.length(input) != math.prod(self.input_shape):
            raise RuntimeError("Invalid input shape for GuptaParata layer")

        output = input @ self.params['weights'] + self.params['biases']

        # Apply activation function
        if self.activation == 'relu':
            output = Tanitra.relu(output)
        elif self.activation == 'sigmoid':
            output = Tanitra.sigmoid(output)
        elif self.activation == 'linear':
            pass
        elif self.activation == 'softmax':
            output = Tanitra.softmax(output)
        else:
            raise RuntimeError("Invalid activation function")
        return output

# ===== Output Layer (NirgamParata) =====
class NirgamParata(GuptaParata):
    # Inherits everything from GuptaParata
    pass

# ===== Normalization Layer =====
class Samasuchaka:
    def __init__(self, normalization_type):
        self.normalization = normalization_type
        if self.normalization == 'z-score':
            self.X_mean = self.X_std = self.y_mean = self.y_std = None
        elif self.normalization == 'min-max':
            self.X_min = self.X_max = self.y_min = self.y_max = None
        else:
            raise ValueError("Choose 'z-score' or 'min-max'")

    def learn(self, X, y):
        if self.normalization == 'z-score':
            self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0)
            self.y_mean, self.y_std = y.mean(axis=0), y.std(axis=0)
        if self.normalization == 'min-max':
            self.X_min, self.X_max = X.min(axis=0), X.max(axis=0)
            self.y_min, self.y_max = y.min(axis=0), y.max(axis=0)

    def forward(self, X, y):
        if self.X_mean is None:
            raise RuntimeError("Call `learn(X, y)` before forward")
        if self.normalization == 'z-score':
            return (X - self.X_mean) / (self.X_std + 1e-8), (y - self.y_mean) / (self.y_std + 1e-8)
        if self.normalization == 'min-max':
            return (X - self.X_min) / (self.X_max - self.X_min + 1e-8), (y - self.y_min) / (self.y_max - self.y_min + 1e-8)

# ===== Convolution Layer =====
class ConvLayer2D:
    def __init__(self, stride, filters, channels, kernel_size, activation,
                 input_shape=None, padding_constant=0, padding_mode=None, padding_width=0):
        self.stride = stride
        self.filters = filters
        self.channels = channels
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.activation = activation
        self.padding = padding_mode
        self.padding_width = padding_width
        self.padding_constant = padding_constant
        self.input_output_learned = False
        self.params = {}

    def forward(self, X):
        if X.shape[0] != self.channels:
            raise ValueError("Channel mismatch")
        output = Tanitra.Tanitra([])
        if not self.input_output_learned:
            self.input_shape = X.shape
            self.output = (self.filters,
                           (self.input_shape[1] - self.kernel_size) // self.stride + 1,
                           (self.input_shape[2] - self.kernel_size) // self.stride + 1)
            for i in range(self.filters):
                self.params['kernels' + str(i)] = Tanitra.Tanitra(
                    cp.random.randn(self.channels, self.kernel_size, self.kernel_size) /
                    (self.input_shape[1] * self.input_shape[2])
                )
            self.input_output_learned = True

        for i in range(self.filters):
            feature_map = Tanitra.Tanitra(0)
            for j in range(self.channels):
                feature_map += Tanitra.convolution2d(
                    X[j], self.params['kernels' + str(i)][j],
                    self.stride, self.padding, self.padding_width, self.padding_constant)
            output = output.append(feature_map)

        # Apply activation
        if self.activation == 'relu':
            output = Tanitra.relu(output)
        elif self.activation == 'sigmoid':
            output = Tanitra.sigmoid(output)
        return output

# ===== MaxPooling Layer =====
class MaxPoolingLayer2D:
    def __init__(self, stride, pool_window, channels, padding_mode=None, pad_width=0, pad_constants=0, input_shape=None):
        self.stride = stride
        self.pool_window = pool_window
        self.channels = channels
        self.padding = padding_mode
        self.pad_width = pad_width
        self.pad_constants = pad_constants
        self.input_output_learned = False
        self.input_shape = None

    def forward(self, X):
        if X.shape[0] != self.channels:
            raise ValueError("Channel mismatch")
        output = Tanitra.Tanitra([])
        if not self.input_output_learned:
            self.input_shape = X.shape
            self.input_output_learned = True
        for j in range(self.channels):
            output = output.append(
                Tanitra.pooling2d(X[j], self.pool_window, self.stride,
                                  self.padding, self.pad_width, self.pad_constants))
        return output

# ===== LSTM Block =====
class LSTM:
    def __init__(self):
        self.long_term_memory = None
        self.short_term_memory = None
        self.params = {}
        self.params_initialized = False

    def forward(self, X):
        if not isinstance(X, Tanitra.Tanitra):
            X = Tanitra.Tanitra(X)
        if not self.params_initialized:
            # Initializing all weights and biases for gates
            for gate in ['forget', 'input', 'input%', 'output']:
                self.params[f'{gate}_gate_short_memory_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
                self.params[f'{gate}_gate_input_weights'] = Tanitra.Tanitra(cp.random.randn(len(X.data), len(X.data)))
                self.params[f'{gate}_gate_biases'] = Tanitra.Tanitra(cp.random.randn(len(X.data)))
            self.params_initialized = True

        self.long_term_memory = Tanitra.Tanitra(cp.zeros_like(X.data))
        self.short_term_memory = Tanitra.Tanitra(cp.zeros_like(X.data))

        for i in range(len(X.data)):
            # Forget gate
            percentage_remember = (self.short_term_memory @ self.params['forget_gate_short_memory_weights'] +
                                   X[i] @ self.params['forget_gate_input_weights'] +
                                   self.params['forget_gate_biases'])
            percentage_remember = Tanitra.sigmoid(percentage_remember)
            self.long_term_memory *= percentage_remember

            # Input gate (new memory)
            potential_memory = (self.short_term_memory @ self.params['input_gate_short_memory_weights'] +
                                X[i] @ self.params['input_gate_input_weights'] +
                                self.params['input_gate_biases'])
            potential_memory = Tanitra.tanh(potential_memory)

            # How much of potential memory to remember
            potential_memory_remember = (self.short_term_memory @ self.params['input_gate%_short_memory_weights'] +
                                         X[i] @ self.params['input_gate%_input_weights'] +
                                         self.params['input_gate%_biases'])
            potential_memory_remember = Tanitra.sigmoid(potential_memory_remember)
            self.long_term_memory += potential_memory * potential_memory_remember

            # Output gate
            percentage_short_term_remember = (self.short_term_memory @ self.params['output_gate_short_memory_weights'] +
                                              X[i] @ self.params['output_gate_input_weights'] +
                                              self.params['output_gate_biases'])
            percentage_short_term_remember = Tanitra.sigmoid(percentage_short_term_remember)
            self.short_term_memory = percentage_short_term_remember * self.long_term_memory

# ===== Self-Attention Layer =====
class SelfAttention:
    def __init__(self, embedding_dim, d_model):
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.params = {
            'Q': Tanitra.Tanitra(cp.random.randn(embedding_dim, d_model) / (embedding_dim ** 0.5)),
            'K': Tanitra.Tanitra(cp.random.randn(embedding_dim, d_model) / (embedding_dim ** 0.5)),
            'V_down': Tanitra.Tanitra(cp.random.randn(embedding_dim, d_model) / (embedding_dim ** 0.5)),
            'V_up': Tanitra.Tanitra(cp.random.randn(d_model, embedding_dim) / (embedding_dim ** 0.5))
        }

    def forward(self, x):
        if not isinstance(x, Tanitra.Tanitra):
            x = Tanitra.Tanitra(x)
        key = x @ self.params['K']
        query = x @ self.params['Q']
        attention = (query @ key.T()) / (self.d_model ** 0.5)
        attention_softmaxed = Tanitra.softmax(attention, axis=1)
        value = x @ self.params['V_down'] @ self.params['V_up']
        y = x + attention_softmaxed @ value
        return y

# ===== Multi-Head Attention Layer =====
class MultiHeadedAttention:
    def __init__(self, embedding_dim, d_model, n_heads):
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_layers = [SelfAttention(embedding_dim, d_model) for _ in range(n_heads)]

    def forward(self, x):
        if not isinstance(x, Tanitra.Tanitra):
            x = Tanitra.Tanitra(x)
        add = Tanitra.Tanitra(cp.zeros_like(x.data))
        for layer in self.attention_layers:
            y = layer.forward(x)
            add += y - x
        x += add
        return x

# ===== Positional Encoding =====
class PositionalEncoding:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def forward(self, x):
        if not isinstance(x, Tanitra.Tanitra):
            x = Tanitra.Tanitra(x)
        if Tanitra.length(x[0]) != self.embedding_dim:
            raise RuntimeError("Embedding dimension mismatch")
        positional_encoding = cp.zeros_like(x.data, dtype=cp.float64)
        for i in range(Tanitra.length(x)):
            for j in range(self.embedding_dim):
                if j % 2 == 0:
                    positional_encoding[i][j] = cp.sin(i / (10000 ** (2 * j / self.embedding_dim)))
                else:
                    positional_encoding[i][j] = cp.cos(i / (10000 ** (2 * (j - 1) / self.embedding_dim)))
        x = x + positional_encoding
        return x

# ===== Final Test for Transformer Components =====
a = Tanitra.Tanitra([[1,2,6,7,3],[6,7,4,8,9],[3,4,4,5,7],[2,4,1,0,8]])
a = PositionalEncoding(5).forward(a)
b = MultiHeadedAttention(5, 2, 5).forward(a)
print(b.data)
