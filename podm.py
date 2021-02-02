import numpy as np
from sklearn.linear_model import LinearRegression
from iforest import IsolationForest

class PoDM:
	def __init__(self, lower_bound=10, higher_bound=13, max_depth=8, random_state=0):
		self.lower_bound = lower_bound
		self.higher_bound = higher_bound
		self.max_depth = max_depth
		self.random_state = random_state

	def fit(self, X):
		np.random.seed(self.random_state)
		self.n_sample = X.shape[0]

		x_arr, y_arr = [], []
		for i in np.arange(self.lower_bound, self.higher_bound):
			sample_size = 2 ** i
			sample = X[np.random.choice(self.n_sample, sample_size, replace=True)]
			clf = IsolationForest(random_state=self.random_state,
								  max_samples=sample_size,
								  contamination='auto').fit(sample, max_depth=100000000)
			depths = np.mean(clf._compute_actual_depth_leaf(sample)[0], axis=0)

			bins = np.arange(int(depths.min()), int(depths.max() + 2))
			y, x = np.histogram(depths, bins=bins)
			y, x = y + 1, x[:-1]
			break_point = np.argmax(y)

			x_arr.append([i])
			y_arr.append(x[break_point])

		self.reg = LinearRegression(fit_intercept=False).fit(x_arr, y_arr)
		self.clf = IsolationForest(random_state=self.random_state,
								   max_samples=len(X),
								   contamination='auto').fit(X, max_depth=self.max_depth)

		return self

	def average_path_length(self, n):
		n = np.array(n)
		apl = self.reg.predict(np.log2([n]).T)
		apl[apl < 1] = 1
		return apl

	def decision_function(self, X):
		depths, leaves = self.clf._compute_actual_depth_leaf(X)

		new_depths = np.zeros(X.shape[0])
		for d, l in zip(depths, leaves):
			new_depths += d + self.average_path_length(l)

		scores = 2 ** (-new_depths
					   / (len(self.clf.estimators_)
						  * self.average_path_length([self.n_sample])))
		return scores
