import numpy as np
import matplotlib.pyplot as plt

file_name = '2-2_data.txt'
nb_points = 250

data = []
with open(file_name, 'r') as file:
	for line in file:
		if line in ['\n','\r\n']: break
		if line[-1] == '\n': line = line[:-1]
		data.append(line)

a = 32 #int(input('Input a: '))
b = 124 #int(input('Input b: '))

dummy = 1e-10

def C(N, M, log=False):
	total = 0
	for i in range(N): total += np.log10(i + 1 + dummy)
	for i in range(M): total -= np.log10(i + 1 + dummy)
	for i in range(N-M): total -= np.log10(i + 1 + dummy)
	if not log: return int(np.around(10 ** total, decimals=1))
	else: return total

def r(a, b, log=False):
	total = 0
	for i in range(1, a+b): total += np.log10(i + dummy)
	for i in range(1, a): total -= np.log10(i + dummy)
	for i in range(1, b): total -= np.log10(i + dummy)

	if not log: return int(np.around(10 ** total, decimals=1))
	else: return total
	

def Beta(p, a, b, log=False):
	if not log: return (p**(a-1)) * ((1-p)**(b-1)) * r(a, b)
	else: return np.log10(p+ dummy)*(a-1) + np.log10(1-p+ dummy)*(b-1) + r(a, b, log = True)

def Binomial(p, N, m, a, b, log=False):
	if not log: return C(N, m) * (p**m) * ((1-p)**(N-m))
	else: return C(N, m, log = True) + m * np.log10(p+ dummy) + (N-m) * np.log10(1-p+ dummy)

def conjugate(p, N, m, a, b):
	return 10 ** (Binomial(p, N, m, a, b, True) + Beta(p, a, b, True))



plt.ion()


X = np.linspace(0, 1, num=nb_points)
for i, line in enumerate(data):
	N = line.count('0') + line.count('1')
	m = line.count('1')
	
	
	Y = [conjugate(x, N, m, a, b) for x in X]
	

	MLE = a/(a+b)
	
	color = [0.55, 0.6 + 0.4/len(data)*(i+1), 0.6 + 0.4/len(data)*(i+1)]
	marginal = (1 / nb_points * sum(Y))

	BinomialLikelihood = 10 ** Binomial(MLE, N, m, a, b, True)
	BetaPrior = 10 ** Beta(MLE, a, b, True)
	lbl =  '%02d, likelihood: %.4f, prior: %.4f, posterior: %.4f' % (i+1, BinomialLikelihood, BetaPrior, np.max(Y)/marginal) #(i+1, np.around(MLE, decimals=2), np.around(X[np.argmax(Y)], decimals=2))

	print('Binomial Likelihood: %.4f, Beta Prior: %.8f, Posterior: %.4f ' % (BinomialLikelihood, BetaPrior, np.max(Y)/marginal))

	a += m
	b += N-m


	x_, y_ = (X[np.argmax(Y)], np.max(Y)/marginal)
	
	
	plt.legend(loc='upper left', prop={'size': 6})
	plt.plot(X, Y/marginal, color=color, label=lbl)
	plt.scatter(x_, y_, color=[1, 0.45, 0.48])
	plt.text( x_, y_, '(%.4f, %.4f)'% (x_, y_), color=[1, 0.48, 0.45])
	plt.show()
	plt.pause(0.25)
		



plt.ioff()
plt.show()


