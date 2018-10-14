import os
import struct
import numpy as np
import cv2

train_label_path = './train-labels-idx1-ubyte'
train_image_path = './train-images-idx3-ubyte'
test_label_path  = './t10k-labels-idx1-ubyte'
test_image_path  = './t10k-images-idx3-ubyte'

def load_mnist():
	with open(train_image_path, 'rb') as img_path:
		m, n, rows, cols = struct.unpack('>IIII', img_path.read(16))
		images = np.fromfile(img_path, dtype=np.uint8)
		train_images = np.reshape(images, (n, 784))
	with open(test_image_path, 'rb') as img_path:
		m, n, rows, cols = struct.unpack('>IIII', img_path.read(16))
		images = np.fromfile(img_path, dtype=np.uint8)
		test_images = np.reshape(images, (n, 784))
	with open(train_label_path, 'rb') as lbl_path:
		m, n = struct.unpack('>II', lbl_path.read(8))
		labels = np.fromfile(lbl_path, dtype=np.uint8)
		train_label = np.reshape(labels, (n, 1))
	with open(test_label_path, 'rb') as lbl_path:
		m, n = struct.unpack('>II', lbl_path.read(8))
		labels = np.fromfile(lbl_path, dtype=np.uint8)
		test_label = np.reshape(labels, (n, 1))
	return train_images, train_label, test_images, test_label



if __name__ == '__main__':
	train_X, train_Y, test_X, test_Y = load_mnist()
	for i in range(0, 10):
		cv2.imshow('%d'%train_labels[i],np.reshape(train_images[i], (28, 28)))
		cv2.waitKey(0)

