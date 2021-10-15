import pandas as pd
import numpy as np

from scipy import stats

def _parallelize(params):
	estimator, X, y, features = params
	estimator.train(X, y)

	return estimator, features

class Rule:
	def __init__(self, id, threshold, categorical=False):
		self.column_id = id
		self.threshold = threshold
		self.categorical = categorical

	def check(self, X):
		if self.categorical:
			return X[:, self.column_id] == self.threshold
		return X[:, self.column_id] >= self.threshold

	def split_train(self, X, y):
		feature = X[self.column_id]

		if self.categorical:
			right_condition = feature == self.threshold
			left_condition = feature != self.threshold
		else:
			right_condition = feature >= self.threshold
			left_condition = feature < self.threshold

		return X[right_condition], \
				y[right_condition], \
				X[left_condition], \
				y[left_condition]

	def split_pred(self, X):
		feature = X[self.column_id]

		if self.categorical:
			return X[feature == self.threshold], X[feature != self.threshold]
		return X[feature >= self.threshold], X[feature < self.threshold]

class Node:
	id=0

	def __init__(self, tree_dict, rule=None, parent_id=None, y=[], curr_depth=None, right_id=None, left_id=None, label=None, max_depth=None, min_n=2, min_split_balance=0.25):
		self.base_id = Node.id
		self.rule = rule
		self.right_id = right_id
		self.left_id = left_id
		self.tree = tree_dict
		self.parent = parent_id
		self.label = label
		self.gini = -1
		self.max_depth = max_depth
		self.min_n = min_n
		self.curr_depth = curr_depth
		self.classes_dist = []
		self.min_split_balance = min_split_balance
		self.gini_index = None

		if len(y) > 0:
			self.gini_index = self._gini(y)

		for label in np.unique(y):
			el = label, len(y[y == label])
			self.classes_dist.append(el)

		Node.id += 1

	def set_left(self, left_id):
		self.left_id = left_id

	def set_right(self, right_id):
		self.right_id = right_id

	def node_gini(self, left, right, size):
		l = len(left) / size
		r = (size - l) / size

		return l * self._gini(left) + r * self._gini(right)

	def _gini(self, y):
		size = len(y)
		coff = 1

		for label in np.unique(y):
			p = len(y[y == label]) / size
			coff -= p ** 2

		return coff

	def create_rules_recursive(self, X, y, curr_depth=0):
		size = len(y)
		best_gini = self._gini(y)
		best_rule = None
		classes = np.unique(y)
		num_cols = X._get_numeric_data().columns

		if (curr_depth >= self.max_depth) or (len(classes) < 2) or (len(y) < self.min_n):
			self.label = stats.mode(y).mode[0]
			return

		for feature in X.columns:
			uniqs = np.unique(X[feature])
			uniqs_sorted = np.sort(uniqs)
			categorical = feature not in num_cols

		for threshold in uniqs_sorted:
			curr_rule = Rule(feature, threshold, categorical=categorical)

			X_right, y_right, X_left, y_left = curr_rule.split_train(X, y)
			curr_gini = self.node_gini(y_left, y_right, size)

			if curr_gini < best_gini:
				best_gini = curr_gini
				best_rule = curr_rule

		if (self._gini(y) <= best_gini) or len(y_right) < 1 or len(y_left) < 1:
			self.label = stats.mode(y).mode[0]
			return

		self.rule = best_rule
		self.gini = best_gini

		X_right, y_right, X_left, y_left = self.rule.split_train(X, y)
		right_node = Node(self.tree, max_depth=self.max_depth, curr_depth=curr_depth+1, y=y_right)
		left_node = Node(self.tree, max_depth=self.max_depth, curr_depth=curr_depth+1, y=y_left)

		self.tree[right_node.base_id] = right_node
		self.tree[left_node.base_id] = left_node

		self.right_id = right_node.base_id
		self.left_id = left_node.base_id

		right_node.create_rules_recursive(X_right, y_right, curr_depth+1)
		left_node.create_rules_recursive(X_left, y_left, curr_depth+1)

	def resolve(self, X, pred):
		if self.label != None:
			pred[X.index] = self.label
			return

		right, left = self.rule.split_pred(X)
		
		self.tree[self.right_id].resolve(right, pred)
		self.tree[self.left_id].resolve(left, pred)

	def print_tree_recurs(self):
		delay = "-" * self.curr_depth

		print("__________________________________")

		if self.label == None:
			print(delay, "id", self.rule.column_id, ">=", self.rule.threshold)

		for label in self.classes_dist:
			print(delay, "class", label[0], "=", label[1])

		print(delay, "Gini index =", self.gini_index)
		print(delay, "Label =", self.label)
		print()

		if self.right_id != None:
			self.tree[self.right_id].print_tree_recurs()

		if self.left_id != None:
			self.tree[self.left_id].print_tree_recurs()
