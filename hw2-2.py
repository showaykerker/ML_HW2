import numpy as np

file_name = '2-2_data.txt'

data = []
with open(file_name, 'r') as file:
	for line in file:
		if line in ['\n','\r\n']: break
		if line[-1] == '\n': line = line[:-1]
		data.append(line)

# a = int(input('Input a: '))
# b = int(input('Input b: '))

def C(N, M):
	total = 0
	for i in range(N): total += np.log10(i+1)
	for i in range(M): total -= np.log10(i+1)
	for i in range(N-M): total -= np.log10(i+1)
	return int(np.around(10 ** total, decimals=1))

def r(n):
	total = 0
	for i in range(1, n): total += np.log10(i)
	return int(np.around(10 ** total, decimals=1))

def Beta(p, a, b):
	return (p**(a-1)) * ((1-p)**(b-1)) * r(a+b) / r(a) / r(b)

print(Beta(0.4, 60, 40))