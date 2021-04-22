import numpy as np
import pandas as pd
import pickle

from scipy.stats import norm
from enum import IntEnum

class Activation(IntEnum):
    Linear = 0
    Sigmoid = 1
    Tanh = 2
    ReLu = 3
    Leaky_ReLu = 4
    PeLu = 5
    ELu = 6
    Arctanh = 7
    Sin = 8
    Gaussian = 9
    Softmax = 10

class Loss(IntEnum):
    ProbabilityPolicyGradient = 0
    BinaryCrossEntropy = 1
    MulticlassCrossEntropy = 2

class GradientOptimizer(IntEnum):
    SGD = 0
    MiniBatch = 1
    Momentum = 2
    Adagrad = 3
    RMSProp = 4
    Adadelta = 5
    Adam = 6

class Model:
    class Target(IntEnum):
        ClassProbability = 0
        PolicyProbability = 1
        Action = 2
        FeatureSpace = 3
        Regression = 4
        Undefined = 5

    def __init__(self, input_size, n_neurons=[], activtions=[], optimizator=None, learning_rate=10**-5, random_state=None):
        if random_state != None:
            np.random.seed(random_state)

        self.input_size = input_size
        self.learning_rate = learning_rate

        self.layers_size = [self.input_size] + n_neurons
        self.activation_pipeline = activtions
        self.last_layer = self.Target.ClassProbability
        self.optimizator = optimizator

        self.activations = [self._linear,
                            self._sigmoid,
                            self._tanh,
                            self._relu,
                            self._leaky_relu,
                            self._pelu,
                            self._elu,
                            self._arctanh,
                            self._sin,
                            self._gaussian,
                            self._softmax]
        
        self.derivatives = [self._linear_derivative,
                            self._sigmoid_derivative,
                            self._tanh_derivative,
                            self._relu_derivative,
                            self._leaky_relu_derivative,
                            self._pelu_derivative,
                            self._elu_derivative,
                            self._arctanh_derivative,
                            self._sin_derivative,
                            self._gaussian_derivative,
                            self._softmax_derivative]

        self.losses = [self._prob_policy_gradient,
                        self._binary_cross_entropy,
                        self._multiclass_cross_entropy]
        
        self.losses_derivatives = [self._prob_policy_gradient_derivative]

        self.pipeline = []
        self.nn = []
        self.bias = []
        self.loss_type = None
        self.loss = None
        self.loss_grad = None
        self.prop_history = []
        self.grad_history = []

#                           Activation functions
#___________________________________________________________________________________________

    def _tanh(self, X):
        x_ = X.astype(np.float128)
        result = (np.exp(x_) - np.exp(x_ * -1)) / (np.exp(x_) + np.exp(x_ * -1))
        
        return result.astype(np.float64)

    def _sigmoid(self, X):
        x_ = X.astype(np.float128)
        result = 1 / (1 + np.exp(x_ * -1))
        
        return result.astype(np.float64)

    def _softmax(self, X):
        x_ = X.astype(np.float128)
        e_x = np.exp(x_ - np.max(x_))
        result = e_x / e_x.sum(axis=0)
        
        return result.astype(np.float64)

    def _relu(self, X):
        return np.maximum(X, 0)

    def _leaky_relu(self, X):
        return np.where(X >= 0, X, X * 0.01)

    def _pelu(self, X, coff):
        return np.where(X >= 0, X, X * coff)

    def _elu(self, X, coff):
        x_ = X.astype(np.float128)
        result = np.where(x_ >= 0, x_, (np.exp(x_) - 1) * coff)
        
        return result.astype(np.float64)

    def _linear(self, X, coff=1.0):
        return X * coff

    def _arctanh(self, X):
        return 1 / np.tan(X)

    def _sin(self, X):
        return np.sin(X)

    def _gaussian(self, X):
        x_ = X.astype(np.float128)
        return np.exp(x_**2 * -1).astype(np.float64)

#                                  Derivatives
#___________________________________________________________________________________________

    def _tanh_derivative(self, X):
        return 1 - (self.tanh(X) ** 2)

    def _sigmoid_derivative(self, X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def _softmax_derivative(self, X, n_gradients=1):
        soft_output = self._softmax(X)
        s = soft_output.reshape(-1,1)
        jacob_grads = np.diagflat(s) - np.dot(s, s.T)

        return jacob_grads[:n_gradients, :]
    
    def _relu_derivative(self, X):
        return np.where(X >= 0, 1, 0)

    def _leaky_relu_derivative(self, X):
        return np.where(X >= 0, 1, 0.01)

    def _pelu_derivative(self, X, coff):
        return np.where(X >= 0, 1, coff)

    def _elu_derivative(self, X, coff):
        return np.where(X >= 0, 1, self._elu(X, coff) + coff)

    def _linear_derivative(self, X):
        return np.ones(X.shape)

    def _arctanh_derivative(self, X):
        return 1 / (X ** 2 + 1)

    def _sin_derivative(self, X):
        return np.cos(X)

    def _gaussian_derivative(self, X):
        x_ = X.astype(np.float128)
        result = np.exp(x_**2 * -1) * -2 * x_
        
        return result.astype(np.float64)

#                                    Loss functions
#___________________________________________________________________________________________

    def _rewards_policy(self, rewards, delta=1.):
        total_rewards = []
        total = .0
        size = len(rewards)

        for i in reversed(range(len(rewards))):
            curr_delta = delta ** (size - i)
            total += rewards[i] * curr_delta

            total_rewards.insert(0,total)

        return np.array(total_rewards)


    def _log_likelihood(self, prob, r):
        prob_size = prob.shape[1]
        r_ = np.vstack([r] * prob_size).T
        
        return np.log(prob) * r_

    def _prob_policy_gradient(self, mean, std, actions, rewards):
        losses = []
        total_rewards = self._rewards_policy(rewards)

        probs = norm.pdf(actions, mean, std)
        losses = self._log_likelihood(probs, total_rewards)

        return np.sum(-losses, axis=0)

    def _binary_cross_entropy(self, y, pred):
        return np.sum(np.where(y == 1, -np.log(pred), -np.log(1 - pred)))

    def _multiclass_cross_entropy(self, y, pred):
        return np.sum( -np.sum(y * np.log(pred), axis=0) )

#                                    Losses derivatives
#___________________________________________________________________________________________

    def _log_likelihood_derivative(self, prob, r):
        prob_size = prob.shape[1]
        r_ = np.vstack([r] * prob_size).T
        
        return r_ / prob

    def _gaussian_dist_prob_derivative(self, x, mean, std):
        return (mean - x) / std ** 2

    def _prob_policy_gradient_derivative(self, mean, std, actions, rewards):
        total_rewards = self._rewards_policy(rewards)

        probs = norm.pdf(actions, mean, std)
        loss_grad = np.nan_to_num(self._log_likelihood_derivative(probs, total_rewards))
        policy_grad = np.nan_to_num(probs * self._gaussian_dist_prob_derivative(actions, mean, std))

        return np.repeat(loss_grad * policy_grad, 2, axis=1)

#                                     Model management
#___________________________________________________________________________________________-


    def add_FC(self, n_neurons, activation):
        self.layers_size.append(n_neurons)
        self.activation_pipeline.append(activation)

    def add_ProbPolicyGradient(self, n_actions, activation):
        self.last_layer = self.Target.PolicyProbability

        self.layers_size.append(n_actions*2)
        self.activation_pipeline.append(activation)

    def add_loss(self, loss_type):
        self.loss_type = loss_type

    def pop(self):
        if len(layer_size) > 1:
            del self.layers_size[-1]
        
        if len(self.activation_pipeline) > 0:
            del self.activation_pipeline[-1]
        
        if len(self.pipeline) > 0:
            del self.pipeline[-1]
        
        if len(self.nn) > 0:
            del self.nn[-1]

        if len(self.bias) > 0:
            del self.bias[-1]

    def save_weights(self, file):
        model = [self.nn, self.bias, self.activation_pipeline]
        pickle.dump(model, open(file, 'wb'))

    def load_weights(self, file):
        model = pickle.load(open(file, 'rb'))

        self.nn = model[0]
        self.bias = model[1]
        self.activation_pipeline = model[2]
        self.pipeline = [self.activations[act] for act in self.activation_pipeline]


#                                        Model general
#___________________________________________________________________________________________


    def init_nn(self):
        self.pipeline = [self.activations[act] for act in self.activation_pipeline]

        if self.loss_type != None:
            self.loss = self.losses[self.loss_type]
            self.loss_grad = self.losses_derivatives[self.loss_type]

        for i in range(len(self.layers_size) - 1):
            input_size = self.layers_size[i]
            output_size = self.layers_size[i+1]

            self.nn.append(np.random.normal(size=(input_size, output_size)))
            self.bias.append(np.random.normal(size=(1, output_size)))

    def prop(self, X, save_history=False):
        result = np.copy(X)

        self.prop_history.append(np.copy(X))

        for layer, bias, activation in zip(self.nn, self.bias, self.pipeline):
            result = activation(np.dot(result, layer) + bias)

            if save_history:
                self.prop_history.append(np.copy(result))

        return result

    def _output_to_mean(self, output):
        size = output.shape[1]
        resized = output.reshape((-1, size // 2, 2))

        return resized[:, :, 0], resized[:, :, 1]

    def _to_jacobian(self, matrix, shape):
        m_dim = matrix.shape[-1]
        h, w = shape

        new_shape = ((-1,) + shape)

        if m_dim == h:
            return np.vstack([matrix] * w).reshape(new_shape)
        if m_dim == w:
            return np.hstack([matrix] * h).reshape(new_shape)

        return matrix

    def _autograds_prob_policy(self, actions, rewards):
        mean, std = self._output_to_mean(self.prop_history[-1])
        nn_grad = []
        bias_grad = []
        self.grad_history = []

        if self.loss_grad == None:
            grad_ = np.ones((1, self.layers_size[-1]))
        else:
            grad_ = self.loss_grad(mean, std, actions, rewards)
            self.grad_history.append(np.copy(grad_))

        for history_i in reversed(range(len(self.prop_history)-1)):
            curr_weights = self.nn[history_i]
            curr_bias = self.bias[history_i]
            curr_shape = curr_weights.shape

            x = self.prop_history[history_i]
            x_ = self._to_jacobian(x, shape=curr_shape)
            out = np.dot(x, curr_weights) + curr_bias
            out_ = self._to_jacobian(out, shape=curr_shape)
            grad_ = self._to_jacobian(grad_, shape=curr_shape)

            grad_ = np.sum(x_ * out_ * grad_, axis=0)

            nn_grad.append(np.copy(grad_))

            grad_ = np.sum(grad_, axis=1)

        return np.array(nn_grad), np.array(bias_grad)

    def predict(self, X):
        if self.last_layer == self.Target.PolicyProbability:
            pred = self.prop(X)
            mean, _ = self._output_to_mean(pred)
            
            return mean
        
        return self.prop(X)
