import cupy as cp
from cupyx.scipy.signal import convolve2d

# Main autodiff class representing tensors with gradient tracking
class Tanitra:

    def __init__(self, data, track_gradient=True):
        self.data = cp.array(data)  # Tensor data (stored as CuPy array)
        self.track_gradient = track_gradient  # Whether to track gradients
        self.parents = []  # List of parent nodes and associated gradient functions
        self.shape = self.data.shape  # Shape of the tensor
        self.grad = None  # Gradient accumulator

    # Element-wise addition
    def __add__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data+other.data)
        if self.track_gradient:
            # d(a + b)/da = 1, d(a + b)/db = 1
            new_tanitra.parents.append((self, lambda g: g*cp.ones_like(other.data)))
            new_tanitra.parents.append((other, lambda g: g*cp.ones_like(self.data)))
        return new_tanitra

    # Element-wise subtraction
    def __sub__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data - other.data)
        if self.track_gradient:
            # d(a - b)/da = 1, d(a - b)/db = -1
            new_tanitra.parents.append((self, lambda g: g* cp.ones_like(other.data)))
            new_tanitra.parents.append((other, lambda g: -g*cp.ones_like(self.data)))
        return new_tanitra

    # Element-wise multiplication
    def __mul__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data * other.data)
        if self.track_gradient:
            # d(a * b)/da = b, d(a * b)/db = a
            new_tanitra.parents.append((self, lambda g : g*other.data))
            new_tanitra.parents.append((other, lambda g: g* self.data))
        return new_tanitra

    # Element-wise division
    def __truediv__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data / other.data)
        if self.track_gradient:
            # d(a / b)/da = 1/b, d(a / b)/db = -a / b^2
            new_tanitra.parents.append((self,lambda g: g * 1/other.data))
            new_tanitra.parents.append((other,lambda g: g * -self.data/(other.data**2)))
        return new_tanitra

    # Matrix multiplication
    def __matmul__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other)
        new_tanitra = Tanitra(self.data @ other.data)
        if self.track_gradient:
            a_shape = self.shape
            b_shape = other.shape
            # Gradient for matmul based on chain rule
            new_tanitra.parents.append((self,lambda g: (g.reshape(-1, b_shape[-1]) @ other.data.T.reshape
            (b_shape[-1],a_shape[-1])).reshape(a_shape)))
            new_tanitra.parents.append((other,lambda g: (self.data.T.reshape(a_shape[-1], -1) @ g.reshape(-1, b_shape[-1
            ])).reshape(b_shape)))
        return new_tanitra

    # Indexing / slicing operation
    def __getitem__(self,index):
        new_tanitra =  Tanitra(self.data[index])
        if self.track_gradient:
            # Backpropagate gradient only to selected index
            def gradn(grad):
                gradient = cp.zeros_like(self.data)
                new_shape = self.data[index].shape
                grad.reshape(new_shape)
                gradient = gradient.astype(cp.float64)
                gradient[index] += grad
                return gradient
            new_tanitra.parents.append((self,gradn))
        return new_tanitra

    # Append another tensor along a specified axis (in-place)
    def add(self,object,axis = 0):
        if not isinstance(object,Tanitra):
            object = Tanitra(object)
        self.data = cp.append(self.data,object.data,axis = axis)

    # Flatten the tensor (preserves shape info for backprop)
    def flatten(self):
        new_tanitra = Tanitra(self.data.flatten())
        if self.track_gradient:
            new_tanitra.parents.append((self, lambda g: g.reshape(self.shape)))
        return new_tanitra

    # Dot product
    def dot(self,a):
        if not isinstance(a,Tanitra):
            a = Tanitra(a)
        new_Tanitra = Tanitra(cp.dot(self.data,a.data))
        new_Tanitra.parents.append((a,lambda g: g*self.data))
        new_Tanitra.parents.append((self, lambda g: g * a))
        return new_Tanitra

    # Append data as nested list (unusual behavior, be cautious)
    def append(self,other):
        if not isinstance(other,Tanitra):
            other = Tanitra(other)
        self_list = self.data.tolist()
        length = len(self_list)
        other_list = other.data.tolist()
        self_list.append(other_list)
        new_tanitra = Tanitra(self_list)
        self.data = cp.array(self_list)
        if self.track_gradient or other.track_gradient:
            new_tanitra.parents.append((self,lambda g: g[:length]))
            new_tanitra.parents.append((other,lambda g: g[-1]))
        return new_tanitra

    # Backward pass to compute gradients
    def backward(self,grad=None):
        if grad is None:
            grad = cp.ones_like(self.data)  # Default gradient for final node
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        for parent,gradient_function in self.parents:
            parent.backward(gradient_function(grad))  # Recursively compute gradient

    # Reset all gradients and graph links
    def grad_0(self):
        for parent,gradient_function in self.parents:
            parent.grad = 0
            parent.grad_0()
        self.parents = []

    # Transpose operation
    def T(self):
        x = Tanitra(self.data.T)
        if self.track_gradient:
            x.parents.append((self,lambda g:g.T))
        return x

# -------- Activation and Utility Functions --------

# Sigmoid activation
def sigmoid(data):
    if not isinstance(data,Tanitra):
        data = Tanitra(data)
    new_tanitra = Tanitra(1 / (1 + cp.exp(-data.data)))
    sig = new_tanitra.data
    if data.track_gradient:
        new_tanitra.parents.append((data,lambda g:g*sig*(1-sig)))
    return new_tanitra

# ReLU activation
def relu(data):
    if not isinstance(data,Tanitra):
        data = Tanitra(data)
    new_tanitra = Tanitra(cp.maximum(0,data.data))
    if data.track_gradient:
        new_tanitra.parents.append((data,lambda g: g*(data.data > 0).astype(float)))
    return new_tanitra

# Return length of Tanitra data
def length(data):
    if not isinstance(data,Tanitra):
        raise TypeError("length function is only for Tanitra class objects.")
    return len(data.data)

# Mean value along axis
def mean(data,axis = None):
    return Tanitra(data.data.mean(axis = axis))

# Element-wise square
def square(data):
    if not isinstance(data,Tanitra):
        data = Tanitra(data)
    new_tanitra = Tanitra(cp.square(data.data))
    if data.track_gradient:
        new_tanitra.parents.append((data, lambda g: g * 2 * data.data))
    return new_tanitra

# Extract raw CuPy data
def to_cons(data):
    return data.data

# -------- Convolution and Pooling Operations --------

# 2D convolution with stride and optional padding
def convolution2d(a,b,stride,padding_mode = None,pad_width = 0,constant_values = 0):
    if padding_mode is not None:
        a_padded= cp.pad(a.data,pad_width = pad_width,mode = padding_mode,constant_values = constant_values)
    else:
        a_padded = a.data
    b_flipped = cp.flip(b.data)
    new_tanitra = Tanitra(convolve2d(b_flipped,a_padded,mode = 'valid')[::stride,::stride])
    if a.track_gradient or b.track_gradient:
        # Gradient w.r.t. input
        def gradient_function_a(grad):
            if stride<1:
                raise RuntimeError("Invalid Stride")
            elif stride == 1:
                grad_a = convolve2d(grad, b_flipped, mode='full')
            else:
                H_out = ((a_padded.shape[0] - b_flipped.shape[0] ) // stride) + 1
                W_out = ((a_padded.shape[0] - b_flipped.shape[0]) // stride) + 1
                H_grad = H_out + (H_out - 1) * (stride - 1)
                W_grad = W_out + (W_out - 1) * (stride - 1)
                unsampled = cp.zeros((H_grad, W_grad))
                unsampled[::stride, ::stride] = grad
                grad_a = convolve2d(unsampled, b_flipped, mode='full')
            if padding_mode is not None:
                if isinstance(pad_width, tuple):
                    grad_a = grad_a[pad_width[0][0]:pad_width[0][0]-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]
                else:
                    grad_a = grad_a[pad_width:-pad_width, pad_width:-pad_width]
            return grad_a
        # Gradient w.r.t. kernel
        def gradient_function_b(grad):
            unsampled = cp.zeros((grad.shape[0]+(grad.shape[0]-1)*(stride-1),(grad.shape[1]+(grad.shape[1]-1)*
                                                                              (stride-1))))
            unsampled[::stride,::stride] = grad
            gradient = convolve2d(a_padded, unsampled, mode='valid')
            return gradient
        new_tanitra.parents.append((a,gradient_function_a))
        new_tanitra.parents.append((b,gradient_function_b))
    return new_tanitra

# 2D max pooling operation
def pooling2d(a,pool_size,stride,padding_mode= None,pad_width = None,constant_values = 0):
    if padding_mode is not None:
        a_padded = cp.pad(a.data,pad_width = pad_width,mode = padding_mode,constant_values = constant_values)
    else:
        a_padded = a.data
    m,n = a_padded.shape
    h = (m-pool_size)//stride + 1
    w = (n-pool_size)//stride + 1
    new_tanitra_data = cp.zeros((h,w))
    indices_list = []
    for i in range(h):
        for j in range(w):
            new_tanitra_data[i,j] = cp.max(a.data[i:i+pool_size,j:j+pool_size])
            index = cp.argmax(a.data[i:i + pool_size, j:j + pool_size])
            index_tupple = (index//pool_size+i*stride,index % pool_size+j*stride)
            indices_list.append(index_tupple)
    new_tanitra = Tanitra(new_tanitra_data)
    if a.track_gradient:
        # Backward pass sets gradient only for max index
        def gradient_function(grad):
            unsampled = cp.zeros_like(a_padded)
            row_indices, col_indices = zip(*indices_list)
            unsampled[row_indices, col_indices] = grad.reshape(unsampled[row_indices, col_indices].shape)
            if padding_mode is not None:
                if isinstance(pad_width,tuple):
                    unsampled = grad[pad_width[0][0]:-pad_width[0][1],pad_width[1][0]:-pad_width[1][1]]
                else:
                    unsampled = grad[pad_width:-pad_width,pad_width:-pad_width]
            return unsampled
        new_tanitra.parents.append((a,gradient_function))
    return new_tanitra

# -------- Math Utilities --------

# Softmax activation (along given axis)
def softmax(a,axis=0):
    if not isinstance(a,Tanitra):
        a = Tanitra(a)
    maxed = a.data - cp.max(a.data,axis = axis,keepdims = True)
    exponent = cp.exp(maxed)
    sum = cp.sum(exponent,axis = axis,keepdims = True)
    result = exponent/sum
    new_tanitra = Tanitra(result)
    if a.track_gradient:
        new_tanitra.parents.append((a,lambda g: g * result * (1 - result)))
    return new_tanitra

# Natural logarithm
def log(x):
    if not isinstance(x,Tanitra):
        x = Tanitra(x)
    x.data = cp.clip(x.data, 1e-9, None)  # Avoid log(0)
    a = Tanitra(cp.log(x.data))
    def grad_func(grad):
        return Tanitra(grad/x.data)
    if x.track_gradient:
        a.parents.append((x,grad_func))
    return a

# Hyperbolic tangent
def tanh(x):
    if not isinstance(x,Tanitra):
        x = Tanitra(x)
    a = Tanitra(cp.tanh(x.data))
    def grad_func(grad):
        return grad*(1-(cp.tanh(x.data)**2))
    if x.track_gradient:
        a.parents.append((x,grad_func))
    return a

# Cosine
def cos(x):
    if not isinstance(x,Tanitra):
        x = Tanitra(x)
    a = Tanitra(cp.cos(x.data))
    def grad_func(grad):
        return grad*-1*cp.sin(x.data)
    if x.track_gradient:
        a.parents.append((x,grad_func))
    return a

# Sine
def sin(x):
    if not isinstance(x,Tanitra):
        x = Tanitra(x)
    a = Tanitra(cp.sin(x.data))
    def grad_func(grad):
        return grad*cp.cos(x.data)
    if x.track_gradient:
        a.parents.append((x,grad_func))
    return a
