import tensorflow as tf 
import numpy as np 
import pickle
import sys

def load(filename, n): # = "corpus.npy"
	print "Loading Data..."
	data = np.load("../data/" + filename)
	print "data =", data.shape

	x = data[:n, 0] # = input text
	y = data[:n, 1] # = response
	x_valid = data[n:, 0] # = input text
	y_valid = data[n:, 1] # = response

	print "x_train =", x.shape
	print "y_train =", y.shape
	print "x_valid =", x_valid.shape
	print "y_valid =", y_valid.shape
	print x[0:3]
	print y[0:3]

	return x, y #, x_valid, y_valid

def shuffle_in_unison(a, b):
	print "Shuffling"
	assert len(a) == len(b)
	np.random.seed(0)
	shuffled_a = np.empty(a.shape, dtype = a.dtype)
	shuffled_b = np.empty(b.shape, dtype = b.dtype)
	permutation = np.random.permutation(len(a))
	for old_index, new_index in enumerate(permutation):
	    shuffled_a[new_index] = a[old_index]
	    shuffled_b[new_index] = b[old_index]
	return shuffled_a, shuffled_b

def one_hot(arr, batch_size, num_steps, vocab_size):
	new_arr = np.zeros((batch_size, num_steps, vocab_size), dtype = int)
	for i in range(batch_size):
		new_arr[i][np.arange(arr.shape[1]), arr[i]] = 1
	return new_arr

def compute_sampling_rate(step):
		#if step > 0: return 2
		if step < self.epoch * self.sampling_id[0]: # all = 1
			return 1
		elif step < self.epoch * self.sampling_id[1]: # 2/3 = 1
			if step % 5 == 0: return 2
			else: return 1
		elif step < self.epoch * self.sampling_id[2]: # 2/3 = 1
			if step % 5 == 0 or step % 5 == 2: return 2
			else: return 1
		elif step < self.epoch * self.sampling_id[3]: # 2/3 = 1
			if step % 5 == 0 or step % 5 == 2: return 1
			else: return 2
		elif step < self.epoch * self.sampling_id[4]:
			if step % 5 == 0: return 1
			else: return 2
		else:
			return 2

def preprocess_testing_data(max_length):
	print "Loading and Preprocessing Testing Data..."
	f = open(sys.argv[2], "r")
	word_id = pickle.load(open("./word_id_dict", "rb"))
	lines = []
	other = ["<u>", "<\u>", "</u>", " v ", " q ", " .f ", " z ", " f ", " u "]
	for i, l in enumerate(f):
		l = l.replace("...", " ")
		l = l.replace(".", " . ")			
		l = l.replace(",", " , ")
		l = l.replace("?", " ? ")
		l = l.replace("!", " ! ")
		l = l.replace(";", " ; ")
		l = l.replace(":", " : ")

		l = l.replace("'", "") # .f, number, <u>, <\u>, ;, xxx"s"
		l = l.replace("\xe2", "")
		l = l.replace("-", " ")
		l = l.replace('"', " ")
		for k in range(len(other)): l = l.replace(other[k], " ")
		l = l.lower()
		l = l.split() # 20?
					
		for j in range(len(l)):
			try: 
				a = int(l[j])
				l[j] = "Number"
			except ValueError: continue
		lines.append(l)
		#print l
	print "testing.len =", len(lines)

	# change word to id
	test = np.zeros([len(lines), max_length], dtype = int)
		
	for i in range(len(lines)):
		# for input message
		if len(lines[i]) > max_length - 2: 
			m = int(len(lines[i]) - max_length - 2)
			lines[i] = lines[i][m:]
			
		n = min(max_length-2, len(lines[i]))
		
		pad = max_length - (n+2)
		test[i][pad] = 1
		for k in range(n):
			try:
				test[i][pad+k+1] = word_id[lines[i][k]]
			except KeyError:
				test[i][pad+k+1] = 3
		test[i][pad+n+1] = 2
		#print test[i]

	print "Finished Loading Testing Data"	
	return test
		

