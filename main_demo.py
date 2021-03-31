#!./venv/bin/python3

import numpy as np

from Kirin import Model, Activation, Loss, Optimizator

if __name__ == "__main__":
	nn = Model(input_size=24, random_state=12732)

	nn.add_FC(n_neurons=128, activation=Activation.ReLu)
	nn.add_FC(n_neurons=64, activation=Activation.ReLu)
	nn.add_FC(n_neurons=16, activation=Activation.ReLu)
	nn.add_ProbPolicyGradient(n_actions=4, activation=Activation.Tanh)
	nn.add_loss(loss_type=Loss.ProbabilityPolicyGradient)

	nn.init_nn()

	actions = np.random.normal(size=(3, 4))
	rewards = np.random.normal(size=(3))

	X = np.random.normal(size=(3, 24))
	y = nn.prop(X, save_history=True)

	grads = nn._autograds_prob_policy(actions, rewards)
	pred = nn.predict(X)

	# print(nn.prop_history)
	print(pred)
	print(grads)
