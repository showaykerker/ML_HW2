from transfer_mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

def _calc_mean(X, Y):
	by_pixel = np.zeros(shape=(10, 784))
	n = np.array([0 for i in range(10)])
	for x, y in zip(X, Y):
		n[y[0]] += 1
		by_pixel[y[0]] = by_pixel[y[0]] + x
	for i in range(10): by_pixel[i] = by_pixel[i] / n[i]
	return n, by_pixel

def _calc_variance(X, Y, total_number, mean):
	sorted_X = np.array([X[Y.flatten()==i] for i in range(10)])
	variance = np.zeros(shape=(10,784,))
	for i, xs in enumerate(sorted_X):
		variance[i] = np.var(xs, axis=0)
	return variance

def get_mean_and_variance(X, Y):
	total_number, mean = _calc_mean(X, Y)
	var = _calc_variance(X, Y, total_number, mean)
	return mean, var

def plot_(mean, var):
	fig, axes = plt.subplots(4, 5)
	for i in range(2):
		for j in range(5):
			axes[i, j].imshow(mean[i*5+j].reshape(28, 28)*8)
	var_ = (var-np.max(var))/(np.max(var)-np.min(var)) * 255
	for i in range(2, 4):
		for j in range(5):
			axes[i, j].imshow(var_[(i-2)*5+j].reshape(28, 28)*8)
	plt.show()

def calculate_log_likelihood_Gaussian(x, mean, variance):
	ans = np.zeros(shape=(10,))
	for c in range(10):
		for x_i, mean_i, var_i in zip(x, mean[c], variance[c]):
			ans[c] += - 0.5 * np.log(2*np.pi*var_i+1e-10) - 0.5 * (x_i-mean_i)**2 / (var_i+1e-10)
	return ans


def main():
	train_X, train_Y, test_X, test_Y = load_mnist()
	prior = np.bincount(train_Y[:, 0]) / train_Y.shape[0]
	mean, variance = get_mean_and_variance(train_X, train_Y)
	right = 0
	wrong = 0
	
	for i, (x, y) in enumerate(zip(test_X, test_Y)):
		log_likelihood = calculate_log_likelihood_Gaussian(x, mean, variance)
		if np.argmax(np.log(prior) + log_likelihood) == y: right += 1
		else: wrong += 1
		print('\r %04d (%5.2f%%), success rate: %5.2f%%' % (i, i/test_Y.shape[0]*100, right/(right+wrong)*100), end='')
		
	plot_(mean, variance)



if __name__ == '__main__':
	main()