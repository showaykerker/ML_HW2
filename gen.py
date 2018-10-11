import numpy as np

n = 12
size = (30, 70)
target = 0.7
file_name = '2-2_data.txt'
data = ''

for i in range(n):
	d_size = int(size[0] + np.around(size[0]*np.random.rand()))
	d = ''
	for j in range(d_size):
		if np.random.rand() > 0.5:
			if np.random.rand()+np.random.normal(0, 0.035) < target: d += '1'
			else: d += '0'
		else:
			if np.random.rand()+np.random.normal(0, 0.035) > target: d += '0'
			else: d += '1'
	d += '\n'
	data += d

with open(file_name, 'w') as f:
	f.write(data)
