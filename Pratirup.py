import Tanitra
import cupy as cp
import matplotlib.pyplot as plt
import Parata

# Sequential Model Class
class AnukramikPratirup:
    def __init__(self, layers=None):
        self.layers = []
        if not layers is None:
            for i in layers:
                self.add(i)

    # Add a layer to the model
    def add(self, layer):
        self.layers.append(layer)

    # Forward pass through all layers
    def estimate(self, X):
        activation = X
        for i in self.layers:
            activation = i.forward(activation)
        return activation

    # Training using gradient descent
    def learn(self, X, y, optimizer='Gradient Descent', epochs=1000, lr=0.1, tol=0.00001):
        if optimizer == 'Gradient Descent':
            loss = 0
            epoch = []
            loss_a = []

            for _ in range(epochs):
                prev_loss = loss
                loss = 0

                # Loop over training data
                for i in range(Tanitra.length(X)):
                    y_pred = self.estimate(X[i])  # forward pass
                    loss += Tanitra.to_cons(Tanitra.mean(Tanitra.square(y_pred - y[i])) / Tanitra.length(X))
                    req_loss = Tanitra.square(y_pred - y[i]) / Tanitra.length(X)
                    req_loss.backward()  # backpropagation

                    # Gradient update for each layer parameter
                    for j in self.layers:
                        for k in j.params:
                            j.params[k].data = j.params[k].data - j.params[k].grad * lr / Tanitra.length(X)

                    req_loss.grad_0()

                # Check for convergence
                if prev_loss - loss < tol and _ != 0:
                    break
                if not (prev_loss - loss < tol and _ != 0) and _ == epochs - 1:
                    print("\n\nError did not converge. Please consider increasing number of epochs.")

                print("Epoch no. ", _)
                epoch.append(_)
                loss_a.append(float(loss))
                print("  loss is ", loss)
                print(" ")

            # Plot loss curve
            plt.plot(epoch, loss_a)
            plt.show()


# Word Embedding Model
class ShabdAyamahPratirup:
    def __init__(self, embedding_dimension, method):
        self.vocabulary = 0
        self.token_list = {}  # maps word to index
        self.embedding_dimension = embedding_dimension
        self.method = method  # 'CBOW' or 'skip gram'
        self.sentences = []
        self.sentence_indices = []
        self.params = {}
        self.layers = [Parata.GuptaParata(self.embedding_dimension, 'linear')]
        self.params_initialized = False
        self.grad = {}

    # Convert text sentences to indices based on vocabulary
    def sentence2indices(self, sentences, token_identification='separate by space', token_list=None):
        vocabulary = 0
        sentence_indices = []
        if token_identification == 'separate by space':
            current_token = ''
            for i in range(len(sentences)):
                sentence_indices_ith = []
                for j in sentences[i]:
                    if j.isupper():
                        j = j.lower()

                    # Check for valid characters (remove punctuation)
                    if (j != ' ' and not 33 <= ord(j) <= 47 or 58 <= ord(j) <= 64 or 
                        91 <= ord(j) <= 96 or 123 <= ord(j) <= 126):
                        current_token += j

                    elif j == ' ':
                        if current_token not in self.token_list:
                            self.token_list[current_token] = vocabulary
                            sentence_indices_ith.append(vocabulary)
                            vocabulary += 1
                        else:
                            sentence_indices_ith.append(self.token_list[current_token])
                        current_token = ''
                
                # Final token after loop
                if current_token not in self.token_list:
                    self.token_list[current_token] = vocabulary
                    sentence_indices_ith.append(vocabulary)
                    vocabulary += 1
                else:
                    sentence_indices_ith.append(self.token_list[current_token])
                current_token = ''
                sentence_indices.append(sentence_indices_ith)

        self.vocabulary += vocabulary
        for i in sentence_indices:
            self.sentence_indices.append(i)
        return sentence_indices

    # Forward pass through embedding layer and softmax (training use)
    def forward_fake(self, x):
        if not self.params_initialized:
            # Initialize embedding weights
            self.params = {
                'embeddings': Tanitra.Tanitra(cp.random.randn(self.vocabulary, self.embedding_dimension)
                                              * (1.0 / cp.sqrt(self.vocabulary))),
                'weight_dash': Tanitra.Tanitra(cp.random.randn(self.embedding_dimension, self.vocabulary)
                                               / cp.sqrt(self.vocabulary / 2))
            }
            self.grad = {
                'embeddings': Tanitra.Tanitra(cp.zeros((self.vocabulary, self.embedding_dimension))),
                'weight_dash': Tanitra.Tanitra(cp.zeros((self.embedding_dimension, self.vocabulary)))
            }
            self.params_initialized = True

        if not isinstance(x, Tanitra.Tanitra):
            x = Tanitra.Tanitra(x)

        y = x @ self.params['embeddings']
        y1 = y @ self.params['weight_dash']
        y2 = Tanitra.softmax(y1)  # softmax for classification
        return y2

    # Forward function for inference: return embedding of token indices
    def forward(self, x):
        return self.params['embeddings'][x]

    # Training embedding model (CBOW or Skip-Gram)
    def learn(self, epochs, lr, tol, window_size, sentences=None):
        if sentences is not None:
            for i in sentences:
                self.sentence_indices.append(i)

        self.layers.append(Parata.GuptaParata(self.vocabulary, 'softmax'))

        training_data_x = []
        training_data_y = []
        epoch = []
        loss_a = []

        # Prepare training data from context windows
        for sentence in self.sentence_indices:
            for j in range(len(sentence) - window_size + 1):
                window = sentence[j:j + window_size]
                for target_pos in range(len(window)):
                    target = window[target_pos]
                    train_x = cp.zeros(self.vocabulary)
                    train_x[target] = 1
                    train_y = cp.zeros(self.vocabulary)
                    for context_pos in range(len(window)):
                        if context_pos != target_pos:
                            context = window[context_pos]
                            train_y[context] = 1
                    if self.method == 'CBOW':
                        training_data_x.append(train_y)
                        training_data_y.append(train_x)
                    elif self.method == 'skip gram':
                        training_data_x.append(train_x)
                        training_data_y.append(train_y / (window_size - 1))

        print(training_data_x)
        print(training_data_y)

        # Training loop
        loss = 0
        for _ in range(epochs):
            prev_loss = loss
            loss = 0

            for i in range(len(training_data_x)):
                y_pred = self.forward_fake(training_data_x[i])  # prediction
                loss += Tanitra.to_cons(Tanitra.mean(Tanitra.square(y_pred - training_data_y[i])) /
                                        len(training_data_x))
                req_loss = Tanitra.square(y_pred - training_data_y[i]) / len(training_data_x)
                req_loss.backward()

                # Accumulate gradients
                for j in self.params:
                    self.grad[j].data += self.params[j].grad / len(training_data_x)

                req_loss.grad_0()

            # Update parameters
            for j in self.params:
                self.params[j] = self.params[j] - self.grad[j] * lr

            print("Epoch no. ", _)
            epoch.append(_)
            loss_a.append(float(loss))
            print("  loss is ", loss)
            print(" ")

            # Early stopping
            if prev_loss - loss < tol and _ != 0:
                break
            if not (prev_loss - loss < tol and _ != 0) and _ == epochs - 1:
                print('loss did not converge. please consider increasing the epochs')

        # Plot loss graph
        plt.plot(epoch, loss_a)
        plt.show()


# Example usage of Sequential model
model = AnukramikPratirup([
    Parata.GuptaParata(4, 'relu'),
    Parata.NirgamParata(1, 'relu')
])

# Input training data
data_x = Tanitra.Tanitra([[0.1, 0.2, 0.3, 0.4]])
data_y = Tanitra.Tanitra([[0.2]])

# Input test data
data_test_x = Tanitra.Tanitra([0.1, 0.2, 0.3, 0.4])
data_test_y = Tanitra.Tanitra([2])

# Train the model
model.learn(data_x, data_y)

# Make prediction
print(model.estimate(data_test_x).data)
