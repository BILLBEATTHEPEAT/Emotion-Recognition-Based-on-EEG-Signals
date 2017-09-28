#  multi-class SVM for EEG analysis in Emotion Recognition with the dataset from SEED http://bcmi.sjtu.edu.cn/~seed/index.html
#  I have combined all the train data into one np.array and all the test data into one np.array to test this model.
#  This is aimed to find a model for all people.
#  Best result: acc of 73.4689
# author JB Huang 2017.9.28

import numpy as np
import pandas as pd
from svmutil import *
from sklearn.preprocessing import StandardScaler

def read_data(f):
	if f == 0:   # train data
		data =  np.load('train_data_array.npy')
		label = np.load('train_label.npy')
		return data, label
	elif f == 1:  #test data
		data =  np.load('test_data_array.npy')
		label = np.load('test_label.npy')
		return data, label

def one_vector(label):
	temp = np.zeros(label.shape[0])
	for i in range(label.shape[0]):
		temp[i] = np.where(label[i] == 1)[0]
	return temp

def main():

	print "loading data"

	train, label = read_data(0)
	print train.shape, label.shape
	label = one_vector(label)
	train = train.reshape(243, -1)
	print train.shape, label.shape

	train = train.tolist()
	label = label.tolist()

	
	print "training the one-class SVM"
	prob_train = svm_problem(label, train)

	param = svm_parameter('-s 0 -t 2 -d 4 -c 10')

	model = svm_train(prob_train, param)



	print "predicting the test data"

	test, label_test = read_data(1)
	label_test = one_vector(label_test)
	print test.shape, label_test.shape
	test = test.reshape(162, -1)
	print test.shape, label_test.shape

	label_test = label_test.tolist()
	test = test.tolist()

	print "predicting"
	p_label, p_acc, p_vals = svm_predict(label_test, test, model, '-b 0')
	p_label1, p_acc1, p_vals1 = svm_predict(label, train, model, '-b 0')

	# # pred = np.zeros(200,)
	# print np.array(p_acc).mean()
	# print np.array(p_acc1).mean()

if __name__ == '__main__':

	main()





