import numpy as np


def uni_disk(n, low=0, high=1):
    r = np.random.uniform(low=low, high=high, size=n)  # radius
    theta = np.random.uniform(low=0, high=2*np.pi, size=n)  # angle
    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)
    return x, y

def sythetic_group_anomaly():
	x1, y1 = uni_disk(100000)
	x1 *= 5
	y1 *= 5

	x2, y2 = uni_disk(1000)
	x2 = x2 * 1.5 + 10
	y2 = y2 * 1.5 + 5

	x3, y3 = uni_disk(2000)
	x3 = x3 * 6 + 3
	y3 = y3 - 10

	x4 = [11, -2, 13, 14]
	y4 = [0, 9, -10, 10]

	x = np.concatenate([x1, x2, x3, x4])
	y = np.concatenate([y1, y2, y3, y4])
	X_norm = np.array([x, y]).T

	return X_norm