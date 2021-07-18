import numpy as np


class DeepNeuralNetwork():
    def __init__(self, X, Y, layers_dims, activation, learning_rate, num_iterations, random_seed = 42, regularization=(False, 0), initialization="he", initialization_mult=0.1, gd_optimization=None, gd_optimization_params=()):
        """
        initializatoin: Defualt-"he", "random"
        initialization_mult: To lower intialization values for "random" initialization. Default - 0.1
        gd_optimization: "sd"-Gradient Descent, "Momentum", "Adams"
        gd_optimization_params: (B1, B2)
        regularization: Default-(False, 0), (Boolean, lambda)

        """
        self.X = X
        self.Y = Y
        self.layers_dims = layers_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.parameters = {}
        self.grads = {}
        self.cost = []
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.regularization = regularization
        self.initialization = initialization
        self.initialization_mult = initialization_mult
        self.gd_optimization =  gd_optimization
        self.gd_optimization_params = gd_optimization_params
        self.v = {} # velocity parameter for Momentum
        self.s = {} # parameters for Adams
        self.regularization_temp = 0

    @classmethod
    def sigmoid(cls, Z):
        """
        Compute the sigmoid of Z

        Arguments:
        Z -- A scaler or numpy array of any size
        Returns:
        A -- sigmoid(Z)
        """

        A = 1/(1 + np.exp(-Z))
        return A

    @classmethod
    def sigmoid_derivative(cls, Z):
        """
        Compute the derivative of sigmoid function

        Arguments:
        Z -- A scaler or numpy array of any size
        Returns:
        s -- derivative of sigmoid(Z)
        """
        s = np.multiply(DeepNeuralNetwork.sigmoid(Z), (1-DeepNeuralNetwork.sigmoid(Z)))
        return s

    @classmethod
    def relu(cls, Z):
        """
        Compute the Relu of Z

        Arguments:
        Z -- A scaler or numpy array of any size
        Returns:
        A -- relu(Z)
        """
        A = np.maximum(0, Z)
        return A

    @classmethod
    def relu_derivative(cls, Z):
        """
        Compute the derivative of relu function

        Arguments:
        Z -- A scaler or numpy array of any size
        Returns:
        S -- derivative of relu(Z)
        """
        S = np.array(Z, copy=True)
        S[S<=0] = 0
        S[S>0] = 1
        return S

    @classmethod
    def initialize_random(cls, layers_dims, initialization_mult, random_seed):
        """
        Arguments: 
        layers_dims -- Python array containing size of each layer

        Returns:
        parameters -- Python dictionary containing Nerual Network parameters
        """
        np.random.seed(random_seed)
        parameters = {}
        L = len(layers_dims)

        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * initialization_mult
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

            assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
            assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))
        
        return parameters

    @classmethod
    def initialize_he(cls, layers_dims, random_seed):
        """
        Arguments: 
        layers_dims -- Python array containing size of each layer

        Returns:
        parameters -- Python dictionary containing Nerual Network parameters
        """
        np.random.seed(random_seed)
        parameters = {}
        L = len(layers_dims)

        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
            
            assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
            assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))
        
        return parameters


    def initialize_parameters(self):
        """
        Arguments:
        dims_layers -- Python array containing the dimensions of each layer in the network.

        Returns:
        parameters -- Python dictionary containing the parameters "W1", "b1", "W2", "b2",....,"WL", "bL"
        Wl -- weight matrix of shape (dims_layers[l], dims_layers[l-1])
        """
        # np.random.seed(42)
        # parameters = {}
        # L = len(self.layers_dims)

        # for l in range(1, L):
        #     self.parameters["W" + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) / np.sqrt(self.layers_dims[l-1])
        #     self.parameters["b" + str(l)] = np.zeros((self.layers_dims[l], 1))

        #     assert(self.parameters["W" + str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
        #     assert(self.parameters["b" + str(l)].shape == (self.layers_dims[l], 1))

        if self.initialization == "random":
            self.parameters = DeepNeuralNetwork.initialize_random(self.layers_dims, self.initialization_mult, self.random_seed)
        else:
            self.parameters = DeepNeuralNetwork.initialize_he(self.layers_dims, self.random_seed)
        # return self.parameters # Dont need to return Initialize parameters

    @classmethod
    def linear_forward (cls, A_prev, W, b):
        """
        Arguments:
        A_prev -- activation from previous layer (or input data): (size of previous layer, number of examples)
        W -- weight matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector: numpy array of shape (size of current layer, 1)

        Returns:
        Z -- the input for activation function, also called pre-activation function
        cache -- Python tuple containing "A_prev", "W", "b"; store for backward pass computation
        """

        Z = W.dot(A_prev) + b
        cache = (A_prev, W, b)

        assert(Z.shape == (W.shape[0], A_prev.shape[1]))

        return Z, cache

    @classmethod
    def linear_activation_forward(cls, A_prev, W, b, activation):
        """
        Implement the forward propogation for the LINEAR->ACTIVATION layer

        Argumnets:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weight matrix: numpy array of shape (size current layer, size of previous layer)
        b -- bias vector: numpy array of shape (size of current layer, 1)
        activation -- the activation to be used in this layer, stored as text string: "sigmoid" or "relu"

        Returns:
        A -- the output of activation function, also called the post activation value
        cache -- a Python tuple containing "linear cache" and "activation cache"; stored for backward pass computation
        """

        if activation == "sigmoid":
            Z, linear_cache = DeepNeuralNetwork.linear_forward(A_prev, W, b)
            activation_cache = Z
            A = DeepNeuralNetwork.sigmoid(Z)

        if activation == "relu":
            Z, linear_cache = DeepNeuralNetwork.linear_forward(A_prev, W, b)
            activation_cache = Z
            A = DeepNeuralNetwork.relu(Z)
        
        assert(A.shape == (W.shape[0], A_prev.shape[1]))
        cache = linear_cache, activation_cache

        return A, cache


    def L_model_forward(self, X, parameters):
        """
        Implement forward propogation for the LINEAR->RELU for L-1 layers and LINEAR->SIGMOID for layer L

        Argumnets:
        self.X -- data, numpy  array of shape(input size, number of examples)
        self.parameters -- dictionary, output of initialize_parameters()

        Returns:
        AL -- last post activation value
        caches -- list of caches containing: every cache of linear_activation_forward()
        """

        caches = []
        A = X
        L = len(self.layers_dims)
        activation = self.activation

        # Implement LINEAR -> RELU upto L-1 layers
        for l in range(1, L-1):
            A_prev = A
            Wl = parameters["W"+ str(l)]
            bl = parameters["b" + str(l)]
            A, cache = DeepNeuralNetwork.linear_activation_forward(A_prev, Wl, bl, activation)
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID for layer L
        WL = parameters["W" + str(L-1)]
        bL = parameters["b" + str(L-1)]
        AL, cache = DeepNeuralNetwork.linear_activation_forward(A, WL, bL, "sigmoid")
        caches.append(cache)

        return AL, caches


    @classmethod
    def l2_regularization(cls, parameters):
        l2_norm = 0
        L = len(parameters) // 2
        for l in range(1, L+1):
            wl = parameters["W" + str(l)]
            l2_norm = l2_norm + np.sum(np.square(wl)) 
        return l2_norm


    def compute_cost(self, AL):
        """
        Implement the cost function

        Arguments:
        AL -- probability vector corresponding to the local predictions, shape (1, number of examples)
        self.Y --  true 'label' vector, shape (1, number of examples)
        Returns:
        cost -- cross-entropy cost
        """
        m = self.Y.shape[1]

        # Compute Loss for AL and Y
        cost = (1/m) * -(np.dot(self.Y, np.log(AL).T) + np.dot(1-self.Y, np.log(1-AL).T))
        
        if self.regularization[0]:
            lambd = self.regularization[1]
            l2_norm = DeepNeuralNetwork.l2_regularization(self.parameters)
            l2_regularization_cost = (1/m)*(lambd/2)*l2_norm
            cost = cost + l2_regularization_cost
        
        cost = np.squeeze(cost)

        assert(cost.shape == ())
        return cost

    @classmethod
    def linear_backward(cls, dZ, cache, regularization):
        """
        Implement linear portion of backward propogation for a sigle layer l

        Arguments:
        dZ -- Gradient of cost with respect to linear output of current layer l
        cache -- tuple of values (A_prev, W, b), output of forward propogation in the current layer l

        Returns:
        dA_prev -- Gradient of cost with respect to activatioon of previous layer l-1, same shape as A_prev
        dW -- Gradient of cost with respect to weight W of current layer l, same shape as W
        db -- Gradient of cost with respect to bias b of current layer l, same shape as b
        """

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        if regularization[0]:
            dW = dW + (regularization[1]/m)*W

        assert(dW.shape == W.shape)
        assert(db.shape == b.shape)
        assert(dA_prev.shape == A_prev.shape)

        return dA_prev, dW, db

    @classmethod
    def linear_activation_backward(cls, dA, cache, activation, regularization):
        """
        Implement backward propogation for LINEAR->ACTIVATION layer

        Arguments:
        dA -- post activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache)
        activation -- the activation to be used in this layer, stored as string "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of cost with respect to activatioon of previous layer l-1, same shape as A_prev
        dW -- Gradient of cost with respect to weight W of current layer l, same shape as W
        db -- Gradient of cost with respect to bias b of current layer l, same shape as b
        """

        linear_cache, activation_cache = cache

        if activation == "relu":
            Z = activation_cache
            dZ = dA * DeepNeuralNetwork.relu_derivative(Z)
            dA_prev, dW, db = DeepNeuralNetwork.linear_backward(dZ, linear_cache, regularization)

        elif activation == "sigmoid":
            Z = activation_cache
            dZ = dA * DeepNeuralNetwork.sigmoid_derivative(Z)
            dA_prev, dW, db =  DeepNeuralNetwork.linear_backward(dZ, linear_cache, regularization)

        return dA_prev, dW, db


    def L_model_backward(self, AL, caches):
        """
        Implement the backward propagation for LINEAR->RELU for L-1 layer and LINEAR->SIGMOID

        Arguments:
        AL -- probability vector, output of forward propagation (L_model_foward())
        self.Y -- true "label vector (contianing 0 or 1)
        caches -- list of caches containing from linear_activation_forward(), each cache contains tuple of (linear_cache, activation_cache)

        Returns:
        grads -- A dictionary with gradients of dA, dW, db
        """

        L = len(caches)
        m = AL.shape[1]
        Y = self.Y.reshape(AL.shape)
        activation = self.activation

        # Initialize the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        self.grads["dA" + str(L)], self.grads["dW" + str(L)], self.grads["db" + str(L)] = DeepNeuralNetwork.linear_activation_backward(dAL, current_cache, "sigmoid", self.regularization)

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = DeepNeuralNetwork.linear_activation_backward(self.grads["dA" + str(l+2)], current_cache, activation, self.regularization)
            self.grads["dA" + str(l+1)] = dA_prev_temp
            self.grads["dW" + str(l+1)] = dW_temp
            self.grads["db" + str(l+1)] = db_temp

        # return self.grads

    
    def update_parameters(self):
        """
        Update parameters using gradient descent 

        Arguments:
        self.parameters -- python dictionary containing parameters
        self.grads -- python dictionary containing gradients
        self.learning_rate -- float value to determine rate of learing
        """
        L = len(self.parameters) // 2
        
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - (self.learning_rate * self.grads["dW" +str(l+1)])
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - (self.learning_rate * self.grads["db" +str(l+1)])



    def build_NN(self):
        np.random.seed(self.random_seed)
        # Initialize parameters
        t = 0
        self.initialize_parameters()
        if self.gd_optimization == "sd":
            pass
        elif self.gd_optimization == "momentum":
            self.v = DeepNeuralNetwork.initialize_velocity(self.parameters)
        elif self.gd_optimization == "adam":
            self.v, self.s = DeepNeuralNetwork.initialize_adam(self.parameters)


        for i in range(0, self.num_iterations):
            AL, caches = self.L_model_forward(self.X, self.parameters) # Forward Propagation
            cost = self.compute_cost(AL) # Compute Cost
            self.L_model_backward(AL, caches) # Backward propagation
            
            # Update Parameters
            if self.gd_optimization == "sd":
                self.update_parameters()
            elif self.gd_optimization == "momentum":
                t = t + 1
                self.v = self.update_parameters_with_momentum(self.v, t=t)
            elif self.gd_optimization == "adam":
                t = t + 1
                self.v, self.s = self.update_parameters_with_adam(self.v, self.s, epsilon=1e-8, t=t)

            # store cost at every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
                self.cost.append(cost)

                        
    def predict(self, X, Y):
        """
        Predicts the results of L-layer model

        Argumnets:
        self.X -- Input matrix
        self.Y --  true 'label' vector, shape (1, number of examples)
        self.parameters -- python dictionary contraining updated parameters

        Returns:
        prediction -- predictions of the given dataset
        """
        m = X.shape[1]
        n = len(self.parameters) // 2
        p = np.zeros((1, m), dtype=int)

        # Forward Propagation
        probas, caches = self.L_model_forward(X=X, parameters=self.parameters)

        # Convert probas into 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print("Accuracy: " + str(np.sum(p == Y)/float(m)))

        return p

    
    def print_cost(self):
        """
        Print cost of training
        
        Arguments: 
        self -- self.cost, self.print_cost, self.learning_rate

        Returns:
        None
        """
        costs = self.cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

    @classmethod
    def initialize_velocity(cls, parameters):
        L = len(parameters) // 2
        v = {}

        for l in range(1, L+1):
            v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        return v

    @classmethod
    def initialize_adam(cls, parameters):
        L = len(parameters) // 2
        v = {}
        s = {}

        for l in range(1, L+1):
            v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
            s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        
        return v, s


    def update_parameters_with_momentum(self, v, t):
        L = len(self.parameters) // 2
        beta1 = self.gd_optimization_params[0]
        v_corrected = {}
        for l in range(1, L+1):
            v["dW" + str(l)] = beta1*v["dW" + str(l)] + (1-beta1)*self.grads["dW" + str(l)]
            v["db" + str(l)] = beta1*v["db" + str(l)] + (1-beta1)*self.grads["db" + str(l)]
            v_corrected["dW" + str(l)] = v["dW" + str(l)]/(1-beta1**t)
            v_corrected["db" + str(l)] = v["db" + str(l)]/(1-beta1**t)

            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate*v_corrected["dW" + str(l)] 
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate*v_corrected["db" + str(l)]
        return v


    def update_parameters_with_adam(self, v, s, epsilon, t):
        L = len(self.parameters) // 2
        beta1 = self.gd_optimization_params[0]
        beta2 = self.gd_optimization_params[1]
        v_corrected = {}
        s_corrected = {}

        for l in range(1, L+1):
            v["dW" + str(l)] = beta1*v["dW" + str(l)] + (1-beta1)*self.grads["dW" + str(l)]
            v["db" + str(l)] = beta1*v["db" + str(l)] + (1-beta1)*self.grads["db" + str(l)]
            s["dW" + str(l)] = beta2*s["dW" + str(l)] + (1-beta2)*(self.grads["dW" + str(l)])**2
            s["db" + str(l)] = beta2*s["db" + str(l)] + (1-beta2)*(self.grads["db" + str(l)])**2

            # Bias correction
            v_corrected["dW" + str(l)] = v["dW" + str(l)]/(1-beta1**t)
            v_corrected["db" + str(l)] = v["db" + str(l)]/(1-beta1**t)
            s_corrected["dW" + str(l)] = s["dW" + str(l)]/(1-beta2**t)
            s_corrected["db" + str(l)] = s["db" + str(l)]/(1-beta2**t)
            
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate*(v_corrected["dW" + str(l)]/(np.sqrt(s_corrected["dW" + str(l)])+epsilon)) 
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate*(v_corrected["db" + str(l)]/(np.sqrt(s_corrected["db" + str(l)])+epsilon)) 
        
        return v, s




    @classmethod
    def dictionary_to_vector(cls, dic):
        dic_values = dic.values()
        for i, values in enumerate(dic_values):
            if i == 0:
                vector = values.flatten()
            else:
                temp = values.flatten()
                vector = np.concatenate((vector, temp))
        return vector
    
    @classmethod
    def vector_to_dictionary(cls, vector, sample_dic):
        dic = {}
        start_index = 0
        for key, value in sample_dic.items():
            end_index = value.size + start_index
            dic[key] = vector[start_index:end_index].reshape(sample_dic[key].shape)
            start_index = end_index

    
    # def gradient_checking(self, epsilon=1e-7):

    #     # Convert parameters and gradients into vector
    #     parameters_vector = DeepNeuralNetwork.dictionary_to_vector(self.parameters)
    #     grad_vector = DeepNeuralNetwork.dictionary_to_vector(self.grads)
    
        
    #     # initialize vectors
    #     num_parameter = parameters_vector.shape[0]
    #     J_plus = np.zeros((num_parameter, 1))
    #     J_minus = np.zeros((num_parameter, 1))
    #     grad_approx = np.zeros((num_parameter, 1))

    #     # Compute gradients
    #     for i in range(num_parameter):
    #         theta_plus = np.copy(parameters_vector)
    #         theta_plus[i] = theta_plus[i] + epsilon
    #         AL_i, _ = self.L_model_forward(self.X, DeepNeuralNetwork.vector_to_dictionary(theta_plus, self.parameters))
    #         J_plus[i] = self.compute_cost(AL_i)

    #         theta_minus = np.copy(parameters_vector)
    #         theta_minus[i] = theta_minus[i] - epsilon
    #         AL_i, _ = self.L_model_forward(self.X, DeepNeuralNetwork.vector_to_dictionary(theta_minus, self.parameters))
    #         J_minus[i] = self.compute_cost(AL_i)

    #         Compuate gradient approx 
    #         grad_approx[i] = (J_plus[i] - J_minus[i])/(2*epsilon)

    #     # Compare gradient_approx to gradients
    #     numerator = np.linalg.norm(grad_approx - grad_vector)
    #     denominator = np.linalg.norm(grad_approx) + np.linalg.norm(grad_vector)
    #     difference = numerator/denominator

    #     if difference > 2e-7:
    #         print("There is a mistake in the backward propagation! difference = " + str(difference))
    #     else:
    #         print("Your backward propagation works perfectly fine! difference = " + str(difference))

    









    
        
    








    