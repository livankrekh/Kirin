import pytest
import numpy as np
from Kirin import Model
from torch.nn.functional import relu, tanh, sigmoid, linear, elu, leaky_relu
from torch import Tensor

class TestActivations:
	@pytest.fixture
	def x1():
		return np.random.rand(100)

	@pytest.fixture
	def x2():
		return np.random.rand(100, 100)

	@pytest.fixture
	def x3():
		return np.random.rand(100, 100, 100)

	@pytest.fixture
	def x4():
		return np.random.rand(100, 100, 100, 100)

	@pytest.fixture
	def _instance():
		return Model(input_size=1)

	def linear_test(x1, x2, x3, x4, _instance):
		for x in [x1, x2, x3, x4]:
			tens = Tensor(x)
			res = linear(tens).numpy()
			kirin_res = _instance._linear(x)

			assert np.all(res == kirin_res)
