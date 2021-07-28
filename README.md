# Kirin
Keras-style deep learning framework for reinforcement learning. This framework designed so that even absolute beginners can make theirs own model for reinforcement learning problems and experiment in this area.

## Features

1. Keras-like model fitting procedure in `Model` class
2. Activation function in `Activation` class. List of implemented activations:
  * Linear
  * Sigmoid
  * Tanh
  * ReLu
  * Leaky_ReLu
  * PeLu
  * ELu
  * Arctanh
  * Sin
  * Gaussian
  * Softmax
3. Different optimizers in `GradientOptimizer` class. List of implemented optimizers:
  * SGD (Stohastic Gradient Descent)
  * MiniBatch
  * Momentum
  * Adagrad
  * RMSProp
  * Adadelta
  * Adam
4. Loss functions in `Loss` class. List of implemented loss functions:
  * ProbabilityPolicyGradient (Policy Gradient)
  * BinaryCrossEntropy
  * MulticlassCrossEntropy
5. Manual `autograd` as part of backpropagation algorithm (example: for Policy Gradient loss function use `Model._autograds_prob_policy()` to get raw gradients)

## Usage

The main class of the framework is `Model`:
```python
Model(input_size,           # - (required) int, state dimensionality
      n_neurons=[],         # - iterable struct of int,
      			    #   lenth of list - number of layers,
      			    #   element - number of neurons in layer
      activtions=[],        # - iterable of Activations,
      			    #   must be the same lenth as n_neurons
      optimizator=None,     # - GradientOptimizer, optimizer of the model
      learning_rate=10**-5, # - float, learning rate
      random_state=None     # - int, random state
      )
```

To add new fully-connected layer in your model use next function:

```python
Model.add_FC(n_neurons,   # - (required) int, number of neurons
             activation   # - (required) Activation,
	     		  #   activation function of the layer
             )
```

To add a loss function use following function:

```python
Model.add_loss(loss_type # - (required) Loss, loss function of the model
               )
```

To make your model learning a Policy Gradient use this:

```python
Model.add_ProbPolicyGradient(n_actions, # - (required) int, shape (number) of actions
                             activation # - (required) Actiovation,
			     		#   activation function for the mean of the model
                             )
```

Train and predict functions:

```python
Model.train(X, # - (required) numpy.array, input data (if classification)
	       #   or current environment and agent state (if Gradient Policy)
            y  # - (required) numpy.array, labels (if classification)
	       #   or rewards (if Gradient Policy)
	    ) 

Model.predict(X # - (required) numpy.array, input data (if classification)
		#   or current environment and agent state (if Gradient Policy)
	      )

```

## Example

```python
from kdl.Kirin import Model, Activation, Loss, GradientOptimizer

if __name__ == "__main__":
	nn = Model(input_size=24, random_state=12732)

	nn.add_FC(n_neurons=128, activation=Activation.Sigmoid)
	nn.add_FC(n_neurons=18, activation=Activation.Sigmoid)
	nn.add_FC(n_neurons=178, activation=Activation.Sigmoid)
	nn.add_FC(n_neurons=178, activation=Activation.Softmax)
	nn.add_ProbPolicyGradient(n_actions=12, activation=Activation.Tanh)
	nn.add_loss(loss_type=Loss.ProbabilityPolicyGradient)

	nn.init_nn()

	actions = np.random.normal(size=(3, 12))
	rewards = np.random.normal(size=(3))

	X = np.random.normal(size=(3, 24))
	y = nn.prop(X, save_history=True)

	grads = nn._autograds_prob_policy(actions, rewards)
	pred = nn.predict(X)
  
  nn.save_weights("./model")
```

