import numpy as np
import matplotlib.pyplot as plt

file_name = '2-2_data.txt'
nb_points = 100

data = []
with open(file_name, 'r') as file:
	for line in file:
		if line in ['\n','\r\n']: break
		if line[-1] == '\n': line = line[:-1]
		data.append(line)

a = 6 #int(input('Input a: '))
b = 14 #int(input('Input b: '))

def C(N, M):
	total = 0
	for i in range(N): total += np.log10(i+1)
	for i in range(M): total -= np.log10(i+1)
	for i in range(N-M): total -= np.log10(i+1)
	return int(np.around(10 ** total, decimals=1))

def r(a, b):
	total = 0
	for i in range(1, a+b): total += np.log10(i)
	for i in range(1, a): total -= np.log10(i)
	for i in range(1, b): total -= np.log10(i)

	return int(np.around(10 ** total, decimals=1))
	

def Beta(p, a, b):
	return (p**(a-1)) * ((1-p)**(b-1)) * r(a, b)

def Binomial(p, N, m, a, b):
	return C(N, m) * (p**m) * ((1-p)**(N-m))

def conjugate(p, N, m, a, b):
	return Binomial(p, N, m, a, b) * Beta(p, a, b)



plt.ion()


X = np.linspace(0, 1, num=nb_points)
for i, line in enumerate(data):
	N = line.count('0') + line.count('1')
	m = line.count('1')
	MLE = m/N
	
	Y = [conjugate(x, N, m, a, b) for x in X]
	a += m
	b += N-m
	
	color = [0.5 + 0.5/len(data) * (i+1), 0.6, 0.3]
	lbl = (i+1, np.around(MLE, decimals=2), np.around(X[np.argmax(Y)], decimals=2))
	marginal = (1 / nb_points * sum(Y))

	print('MLE: %.4f, Binomial Likelihood: %.4f, Beta Prior: %.4f, Posterior: %.4f ' % (MLE, Binomial(MLE, N, m, a, b), Beta(MLE, a, b), np.max(Y)/marginal/100))
	plt.plot(X, Y/marginal/100, color=color, label=lbl)
	plt.legend(loc='upper left')
	plt.show()
	plt.pause(0.5)

plt.ioff()
plt.show()


