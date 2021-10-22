#!./venv/bin/python3

import numpy as np

from Kirin import Model, Activation, Loss, GradientOptimizer

if __name__ == "__main__":
	nn = Model(input_size=24, random_state=12732)

	nn.add_FC(n_neurons=128, activation=Activation.Sigmoid)
	nn.add_FC(n_neurons=18, activation=Activation.Sigmoid)
	nn.add_FC(n_neurons=178, activation=Activation.Sigmoid)
	nn.add_FC(n_neurons=178, activation=Activation.Softmax)
	# nn.add_ProbPolicyGradient(n_actions=12, activation=Activation.Tanh)
	# nn.add_loss(loss_type=Loss.ProbabilityPolicyGradient)

	nn.init_nn()

	actions = np.random.normal(size=(3, 12))
	rewards = np.random.normal(size=(3))

	X = np.random.normal(size=(3, 24))
	y = nn.prop(X, save_history=True)

	grads = nn._autograds_prob_policy(actions, rewards)
	pred = nn.predict(X)

	# print(nn.prop_history)
	print(pred)
	print(grads)
