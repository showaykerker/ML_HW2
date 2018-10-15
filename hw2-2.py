import numpy as np
import matplotlib.pyplot as plt
import sys


a = 120 #int(input('Input a: '))
b = 580 #int(input('Input b: '))

file_name = '2-2_data.txt'
nb_points = 500
sys.setrecursionlimit(50000)

data = []
with open(file_name, 'r') as file:
	for line in file:
		if line in ['\n','\r\n']: break
		if line[-1] == '\n': line = line[:-1]
		data.append(line)

r_table = [None, None, 0]

dummy = 1e-10


def C(N, M):
	total = log_r(N+1) - log_r(M+1) - log_r(N-M+1)
	return total

def log_r(a):
	if a == 0 or a == 1: raise ValueError()
	if len(r_table) > a: return r_table[a]
	ans = np.log10(a-1) + log_r(a-1)
	r_table.append(ans)
	return ans

def Beta(p, a, b):
	return np.log10(p + dummy)*(a-1) + np.log10(1-p + dummy)*(b-1) + log_r(a+b) - log_r(a) - log_r(b)

def Binomial(p, N, m, a, b):
	return C(N, m) + m * np.log10(p + dummy) + (N-m) * np.log10(1 - p + dummy)

def conjugate(p, N, m, a, b):
	return 10 ** (Binomial(p, N, m, a, b) + Beta(p, a, b))



plt.ion()


X = np.linspace(0, 1, num=nb_points)

for i, line in enumerate(data):
	

	N = line.count('0') + line.count('1')
	m = line.count('1')
	
	MLE = a/(a+b)
	
	Y = [conjugate(x, N, m, a, b) for x in X]
	
	color = [0.55, 0.6 + 0.4/len(data)*(i+1), 0.6 + 0.4/len(data)*(i+1)]
	
	marginal = (1 / nb_points * sum(Y))

	BinomialLikelihood = 10 ** Binomial(MLE, N, m, a, b)
	BetaPrior = 10 ** Beta(MLE, a, b)
	lbl =  '%02d, likelihood: %.4f, prior: %.4f, posterior: %.4f' % (i+1, BinomialLikelihood, BetaPrior, np.max(Y)/marginal) #(i+1, np.around(MLE, decimals=2), np.around(X[np.argmax(Y)], decimals=2))
	print('%03d | Binomial Likelihood: %.8f, Beta Prior: %.8f, Posterior: %.4f ' % ( i+1, BinomialLikelihood, BetaPrior, np.max(Y)/marginal))

	a += m
	b += N-m

	x_, y_ = (X[np.argmax(Y)], np.max(Y)/marginal)	
	
	plt.plot(X, Y/marginal, color=color, label=lbl)
	plt.legend(loc='upper left', prop={'size': 6})
	plt.scatter(x_, y_, color=[1, 0.45, 0.48])
	plt.text( x_, y_, '(%.4f, %.4f)'% (x_, y_), color=[1, 0.48, 0.45])
	plt.show()
	plt.pause(0.01)
		



plt.ioff()
plt.show()


