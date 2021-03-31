import pandas as pd
import numpy as np
import random

from scipy import stats
from multiprocessing import Pool

from tools.tree_tools import Node, Rule, _parallelize

class DecisionTree:
	def __init__(self, max_depth=100):
		self.curr_id = 0
		self.tree = dict()
		self.initial_node = None
		self.labels = []
		self.max_depth = max_depth
		self.X = None

	def train(self, X_raw, y_raw):
		self.labels = list(np.unique(y_raw))
		y = pd.Series([ self.labels.index(x) for x in y_raw ])
		X = pd.DataFrame(X_raw)

		self.initial_node = Node(self.tree, max_depth=self.max_depth, curr_depth=0)
		self.tree[0] = self.initial_node
		self.initial_node.create_rules_recursive(X, y)

	def predict(self, X_raw):
		y = []
		X = pd.DataFrame(X_raw)
		pred = np.array([None] * len(X))

		self.initial_node.resolve(X, pred)

		for x in pred:
			ans = self.labels[x] if x != None else None
			y.append(ans)

		return np.array(y)

	def print_tree(self):
		if self.initial_node != None:
			self.initial_node.print_tree_recurs()
		else:
			print("No trained tree!!!")

class RandomForest:
	def __init__(self, n_estimators=100, max_depth=50, random_state=None, n_jobs=8):
		self.depth = max_depth
		self.n = n_estimators
		self.features_split = []
		self.estimators = [DecisionTree(max_depth=max_depth)] * n_estimators
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.trained = None
		self.classes = []

		if random_state != None:
			random.seed(random_state)

	def train(self, X, y):
		self.classes = np.unique(y)
		parameters = []
		X = pd.DataFrame(X)
		features = X.columns
		n_randoms = int(len(features) ** 0.5)

		for estimator in self.estimators:
			curr_features = random.sample(list(features), n_randoms)
			curr_X = X[curr_features]

			parameters.append((estimator, curr_X, y, curr_features))
			self.features_split.append(curr_features)

		with Pool(self.n_jobs) as p:
			self.trained = p.map(_parallelize, parameters)

	def _predict_raw(self, X):
		preds = []

		for estimator, features in self.trained:
			X_curr = X[features]
			pred = estimator.predict(X_curr)

			preds.append(pred)

		return np.array(preds)

	def predict_proba(self, X):
		X = pd.DataFrame(X)
		y_raw = self._predict_raw(X)
		y = {}

		for cl in self.classes:
			cl_predicts = []

			for n_pred in y_raw:
				prob = len(n_pred[n_pred == cl]) / len(n_pred)
				cl_predicts.append(prob)

			y[cl] = cl_predicts

		return pd.DataFrame(y)

	def predict(self, X):
		X_pd = pd.DataFrame(X)
		y_raw = self._predict_raw(X_pd)
		y = stats.mode(y_raw).mode[0]

		return y
