import tensorflow as tf
import pickle
import numpy as np
import sys

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config = config))

#import matplotlib.pyplot as plt

class Model(object):
	def __init__(self):
		self.data_size = 220000 # 437600
		self.batch_size = 200
		self.num_steps = 30 # n of input layer 1
		self.dim_size = 11792 # n of input layer 2
		self.n_layers = 2 # 2
		self.hidden_size1 = 800 # unknown, need to be ckeck, 200, 650, 1500
		self.hidden_size2 = 1800
		self.hidden_size3 = 2400
		self.vocab_size = 3591 # n of output layer (class)
		self.is_train = 210000 # 420000
		self.v1 = 420000
		self.v2 = 430000
		self.t1 = 200000
		self.batch_num = self.is_train / self.batch_size # how many batch, = 2100
		self.valid_num = (self.data_size - self.is_train) / self.batch_size
		self.valid_size = self.data_size - self.is_train
						# how many validation batch
	
		self.from_last = 1
		self.beta1 = 0.00
		self.beta2 = 0.00
		self.drop = 0.4
		self.lr = 10e-5 # 0.0001~0.001, 0.00003, 3e-4(begin)
		self.epoch = 2201
		self.small = 0
		self.min_loss_v = 1.0
		self.n = 1040

	def shuffle_in_unison(self, a, b):
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

	def lstm_layer_multi(self, X): # dropout set at in/ out of rnn!!!
		X = tf.reshape(X, [-1, self.dim_size])
		X_in = (tf.matmul(X, self.w1) + self.b1)
		#X_in = tf.nn.relu(tf.matmul(X_in, self.w2) + self.b2)
		X_in = tf.reshape(X_in, [-1, self.num_steps, self.hidden_size1])
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size1, forget_bias = 1.0, state_is_tuple = True)
		lstm_stack = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.n_layers, state_is_tuple = True)
		init_state = lstm_stack.zero_state(self.bs, dtype = tf.float32)
		outputs, final_state = tf.nn.dynamic_rnn(lstm_stack, X_in, initial_state = init_state, time_major = False)
		outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2])) # states is the last outputs
		results = tf.matmul(outputs[-1], self.w4) + self.b4
		return results
	
	def set_nn_parameter(self):
  		print "Setting parameter"
  		self.w1 = tf.Variable(tf.random_normal([self.dim_size, self.hidden_size1]))
		self.b1 = tf.Variable(tf.constant(0.1, shape = [self.hidden_size1, ]))

		self.w2 = tf.Variable(tf.random_normal([self.hidden_size1, self.hidden_size2]))
		self.b2 = tf.Variable(tf.constant(0.1, shape = [self.hidden_size2, ]))

		self.w3 = tf.Variable(tf.random_normal([self.hidden_size2, self.hidden_size3]))
		self.b3 = tf.Variable(tf.constant(0.1, shape = [self.hidden_size3, ]))

		self.w4 = tf.Variable(tf.random_normal([self.hidden_size1, self.vocab_size]))
		self.b4 = tf.Variable(tf.constant(0.1, shape = [self.vocab_size, ]))
		self.x_nn = tf.placeholder(tf.float32, [None, self.num_steps, self.dim_size])
		self.y_nn = tf.placeholder(tf.float32, [None, self.vocab_size])
		self.keep_prob = tf.placeholder("float")
		self.bs = tf.placeholder(dtype = tf.int32)

	def cut_batch(self, step):
		self.batch_id = step % self.batch_num
		begin = self.batch_size * self.batch_id
		end = self.batch_size * (self.batch_id + 1)
		return begin, end

	def hotx(self, arr):
		new_arr = np.zeros((1, 30, self.dim_size), dtype = int)
		new_arr[0][np.arange(arr.shape[1]), arr[0]] = 1
		return new_arr

	def hoty(self, arr):
		new_arr = np.zeros((self.batch_size, self.vocab_size))
		new_arr[np.arange(arr.shape[0]), arr] = 1
		return new_arr

	def output_accuracy(self, loss, accuracy):
		acc2 = 0
		loss2 = 0
		n = 70
		for i in range(self.batch_num/n):
			begin = self.batch_size * n*i
			end = self.batch_size * (n*i + 1)
			temp = self.sess.run(accuracy, feed_dict = {self.x_nn: self.hotx(self.x[begin:end]), 
				self.y_nn: self.hoty(self.y_train[begin:end]), self.bs: self.batch_size, self.keep_prob: self.drop}),
			acc2 += temp[0]
			loss2 += self.sess.run(loss, feed_dict = {self.x_nn: self.hotx(self.x[begin:end]), 
				self.y_nn: self.hoty(self.y_train[begin:end]),
				self.bs: self.batch_size, self.keep_prob: self.drop})
			
		acc2 /= (self.batch_num/n)
		loss2 /= (self.batch_num/n)
	
		print ", acc: %.3f" % acc2,
		print ", loss: %.4f" % (loss2 / self.batch_size),
		
		v_acc = 0
		v_loss = 0
		m = 6
		for i in range(self.valid_num/m):	
			n1 = m*i*self.batch_size
			n2 = (m*i+1)*self.batch_size
			v_acc += self.sess.run(accuracy, feed_dict = {self.x_nn: self.hotx(self.x_valid[n1:n2]), 
				self.y_nn: self.hoty(self.y_valid[n1:n2]), self.bs: self.batch_size, self.keep_prob: 1}) 
			v_loss += self.sess.run(loss, feed_dict = {self.x_nn: self.hotx(self.x_valid[n1:n2]), 
				self.y_nn: self.hoty(self.y_valid[n1:n2]), self.bs: self.batch_size, self.keep_prob: 1}) 
			#v_acc += sess.run(accuracy, feed_dict = {x_nn: x_valid, y_nn: y_valid,}) 
		v_acc /= (self.valid_num/m)
		v_loss /= (self.valid_size/m)
		
		print ", v_acc: %.3f" % v_acc,
		print ", v_loss: %.4f" % v_loss
		
	def construct_id_w(self):
		common = pickle.load(open(self.path + "model/common1_word", "rb"))
		self.id_w = {}
		for i in range(self.vocab_size):
			self.id_w[i] = common[i]
		print len(self.id_w)
		for i in range(10):
			print self.id_w[i], 
		pickle.dump(self.id_w, open(self.path + "model/dict_id_w", "wb"))

	def get_ans_id(self, n):
		if n == 0: return "a"
		elif n == 1: return "b"
		elif n == 2: return "c"
		elif n == 3: return "d"
		elif n == 4: return "e"
		else: 
			return "error!"

	def output_answer(self, prob):
		fout = open(sys.argv[1], "wb")
		fout.write("id,answer\n") 
		
		for i in range(self.n):
			m = prob[i].argmax(axis = 0)
			ans = self.get_ans_id(m)
			fout.write("%d,%s\n" %(i+1, ans))
			

	def load_test(self):
		print "Loading data"
		self.temp = pickle.load(open("x_test_hot", "rb"))
		self.y_test = pickle.load(open("y_test_hot", "rb"))
		#self.y_test_id = pickle.load(open(self.path + "model/y_test_id", "rb"))
		self.x_test = np.empty([1040, 30], dtype = int)
		for i in range(self.n):
			for j in range(30):
				kkk = int(self.temp[i][j])
				#if i == 0: print kkk
				self.x_test[i][j] = kkk
		
	def compute_answer(self, sess, pred):
		self.load_test()
		print "Start testing"
		back = 2 # consider the prob of latter 2 words
		
		prob = np.empty([self.n, 5], dtype = float)
		x_t1 = np.empty([1, 30], dtype = int)
		y_t1_id = np.empty(5, dtype = int)
		for i in range(self.n):
			x_t1[0] = self.x_test[i] # for only one testing data
			for j in range(5):
				y_t1_id[j] = int(self.y_test[i][j])
		
			ans = sess.run(pred, feed_dict = {self.x_nn: self.hotx(x_t1), self.bs: 1, self.keep_prob: 1}) 
			for j in range(5):
				prob[i][j] = ans[0][y_t1_id[j]] # input 5 probable y id

			if i%100 == 0: print i
		'''for i in range(5):
			for j in range(5):
				print "%.3f" % prob[i][j],
			print ""'''

		self.output_answer(prob)

	def nn(self, first_time):
		print "Constructing lstm-nn model"
		for iii in range(1):
			self.sess = tf.Session()
			pred = self.lstm_layer_multi(self.x_nn)
			loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_nn, logits = pred)))
			#+ tf.nn.l2_loss(self.w1)*self.beta1 + tf.nn.l2_loss(self.w4)*self.beta2)
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
			optimizer = tf.train.AdamOptimizer(self.lr)
			train_step = optimizer.apply_gradients(zip(grads, tvars))
			correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y_nn, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

			init = tf.global_variables_initializer()
			saver = tf.train.Saver()
		
			#init = tf.global_variables_initializer()
			if self.from_last == 1:
				saver.restore(self.sess, self.path + "/model/model.ckpt")
		  		print("Model restored.")
		  	else:
				self.sess.run(init)
				
			self.step = 0
			batch_id = 0
			
			#while step * self.batch_size < training_iters:
			print "\nStart training"
			flag = 1
			for i in range(1):
				while self.step < self.epoch:
					if self.small == 0:
						begin, end = self.cut_batch(self.step)
						x_batch = self.x[begin:end]
						y_batch = self.y_train[begin:end]

					self.sess.run([train_step], feed_dict = {self.x_nn: self.hotx(x_batch), self.y_nn: self.hoty(y_batch), 
						self.bs: self.batch_size, self.keep_prob: self.drop})
					
					if self.step % 150 == 0:
						print self.step, self.batch_id,
						self.output_accuracy(loss, accuracy)
					
					if self.step == 2000 or self.step == 2100:
						self.lr *= 0.3
						print "lr = ", self.lr
					self.step += 1
				
				save_path = saver.save(self.sess, self.path + "/model/model.ckpt")
	  			print("Model saved in file: %s" % save_path)
	  
  	def test_nn(self):
  		pred = self.lstm_layer_multi(self.x_nn)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_nn, logits = pred))
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
		optimizer = tf.train.AdamOptimizer(self.lr)
		train_step = optimizer.apply_gradients(zip(grads, tvars))
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y_nn, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess, "hw1/testing_env/model/model.ckpt")
		  	print("Model restored")
  			self.compute_answer(sess, pred)

model = Model()
model.set_nn_parameter()
model.test_nn()
print "\n Compile successfully!"

