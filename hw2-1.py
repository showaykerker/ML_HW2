from transfer_mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

class NaiveBayes:
	def __init__(self, mode='continuous', smooth=1):
		self.mode = mode
		self.smooth = 3900 if mode == 'continuous' else 1e-4
		
		

	def fit(self, X, Y):
		if self.mode == 'continuous': self.X, self.Y = X.copy(), Y.flatten()
		else: self.X, self.Y = np.floor(X.copy()/8), Y.flatten()
		self.n_data, self.n_features = self.Y.shape[0], self.X.shape[1]
		self.n_classes = len(np.bincount(self.Y))
		self.mu, self.var = self._get_mu_and_var()
		self.prior = np.bincount( self.Y ) / self.Y.shape[0]
		self.bincount = np.bincount( self.Y )
		if self.mode != 'continuous': self.freq = self._tally_frequency()
		
		print('Fit done.')

	def _get_mu_and_var(self):
		mu = np.zeros(shape=(self.n_classes, self.n_features))
		var  = np.zeros(shape=(self.n_classes, self.n_features))
		for c in range(self.n_classes): 
			curr_X = self.X[self.Y==c]
			mu[c] = curr_X.mean(axis=0)
			var[c] = curr_X.var(axis=0) + self.smooth
		return mu, var

	def _tally_frequency(self):
		tally = np.zeros(shape=(self.n_classes, self.n_features, 32))
		for c in range(self.n_classes):
			curr_X = np.floor(self.X[self.Y==c]/8).astype(np.uint8)
			for i in range(784):
				unique, counts = (np.unique(curr_X[:, i], return_counts=1))
				tally[c][i][unique]=counts
			tally[c] = tally[c] / curr_X.shape[0]
			tally[c] += self.smooth
		return tally

	def _get_log_likelihood_discrete(self, x):
		x = np.floor(x/8)
		log_likelihood = np.zeros(shape=(self.n_classes), dtype=np.float32)
		for i, f in enumerate(x):
			log_likelihood += np.log(self.freq[:, i, int(f)])
		return log_likelihood

	def _get_log_likelihood(self, x):
		log_likelihood = np.zeros(shape=(self.n_classes), dtype=np.float32)
		for c in range(self.n_classes):
			tmp = np.array( - 0.5 * np.log( 2 * np.pi * self.var[c] ) - 0.5 * ( x - self.mu[c] ) ** 2 / self.var[c] )
			log_likelihood[c] = sum(tmp.flatten())
		return log_likelihood			

	def plot(self):

		fig, axes = plt.subplots(4, 5)
		for i in range(2):
			for j in range(5):
				axes[i, j].imshow(self.mu[i*5+j].reshape(28, 28)*8)
		var_ = (self.var-np.max(self.var))/(np.max(self.var)-np.min(self.var)) * 255
		for i in range(2, 4):
			for j in range(5):
				axes[i, j].imshow(var_[(i-2)*5+j].reshape(28, 28)*8)
		fig2, axes2 = plt.subplots(2, sharex=True)
		
		for c in range(self.n_classes):
			axes2[0].plot([i for i in range(self.n_classes)], self.errors[c][:-3], label=str(c), linestyle='--')

		axes2[1].plot([i for i in range(self.n_classes)], self.errors[:, -1], linestyle='--')
		axes2[0].legend(loc='best')
		plt.xticks(np.linspace(-1, 10, 12))

		plt.show()

	def score(self, X, Y):
		predict = self.predict(X)
		right = 1
		self.errors = np.zeros(shape=(self.n_classes, self.n_classes+3), dtype=np.float16)
		for p, y in zip(predict, Y.flatten()): 
			if p == y : right += 1
			else: self.errors[y][p] += 1
		#print(self.errors[:, :-2])
		#print(np.sum(self.errors[:, :-2], axis=1))
		self.errors[:, -1] = np.sum(self.errors[:, :-3], axis=1) / np.bincount(Y.flatten())
		self.errors[:, -2] = np.bincount(Y.flatten())
		self.errors[:, -3] = np.sum(self.errors[:, :-3], axis=1)
		#for c in range(self.n_classes): self.errors[c][-1] = np.sum(self.errors[c, :-1])/np.bincount(Y)[c]
		total = Y.flatten().shape[0]
		wrong = total - right
		return right/total, wrong/total, self.errors

	def predict(self, X):
		ans = []
		for x in X: 
			if self.mode == 'continuous': log_likelihood = self._get_log_likelihood(x)
			else : log_likelihood = self._get_log_likelihood_discrete(x)
			ans.append(np.argmax(np.log(self.prior) + log_likelihood))
		return np.array(ans)



def main(mode='continuous'):
	train_X, train_Y, test_X, test_Y = load_mnist()
	a = NaiveBayes(mode=mode)
	a.fit(train_X, train_Y)
	right, wrong, errors = a.score(test_X, test_Y)
	print('Error Rate:', wrong)
	print('Error Rate (number-wise)', errors[:, -1])
	print(errors[:, :-1].astype(np.uint32))
	a.plot()


	

def test(mode='continuous'):
	train_X, train_Y, test_X, test_Y = load_mnist()
	outcome = []
	smooth_ = []
	for smooth in np.linspace(1e-6, 1e-4, 10):
		print('smooth:', smooth, end = ' --> ')
		a = NaiveBayes(mode=mode, smooth=smooth)
		a.fit(train_X, train_Y)
		right, wrong, errors = a.score(test_X, test_Y)
		print(wrong)
		smooth_.append(smooth)
		outcome.append(wrong)
	#plt.plot(smooth_, outcome, 'o-')
	#plt.show()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', '-m', dest='mode', action='store', default=0, help='0: discrete, 1: continuous mode')
	args = parser.parse_args()
	args.mode = int(args.mode)

	if args.mode == 0: 
		print('Discrete Mode.')
		main('discrete')
	else: 
		print('Continuous Mode.')
		main('continuous')

