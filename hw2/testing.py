import tensorflow as tf
import numpy as np
import pickle
import sys
import random

from keras.backend.tensorflow_backend import set_session
#if 'session' in locals() and session is not None:
#    print('Close interactive session')
#    session.close()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config = config))

#best

class Model(object):
	def __init__(self):
		self.first_epoch = 0
		self.epoch = 29 * 0.1300
		self.drop = 1
		self.lr = 1e-3 # 0.0001~0.001, 0.00003, 3e-4(begin)

		self.data_size = 1450
		self.batch_size = 50
		self.num_steps = 80 # n of input layer 1
		self.num_steps_out = 42 + 2 # n of input layer 1
		self.dim_size = 4096 # n of input layer 2
		self.n_layers = 1 # 
		self.hidden1_size = 300 #300
		self.w_size = 300 #300
		self.hidden2_size = 400
		self.h_time = 10
		self.n = 80/self.h_time
		
		self.vocab_size = 5993 + 2 # n of output layer (class)
		self.is_train = 1400
		self.EOS_id = self.vocab_size - 1 # 5994
	
		self.batch_num = self.is_train / self.batch_size # num of train batch
		self.valid_num = 1
		self.valid_size = self.data_size - self.is_train # num of valid
		self.name = str(self.hidden1_size) + "-" + str(self.hidden2_size)
		self.fout = open('terminalnew' + self.name,'w')
		
	'''def process_y(self):
		#self.y = np.load("../model/all_label_preprocess_idx")
		self.y = np.load("../model/all_label_select_id")
		for i in range(self.data_size):
			for j in range(len(self.y[i])):
				self.y[i][j] = self.y[i][j] + [self.EOS_id] * (self.num_steps_out - len(self.y[i][j]))
				self.y[i][j] = np.array(self.y[i][j])
		# = 1450, 5~37, x~44'''

	'''def load(self):
		self.x = np.load("../model/all_feat.npy")
		self.process_y()

		self.x_valid = self.x[self.is_train:]
		self.y_valid = self.y[self.is_train:]
		self.x = self.x[:self.is_train]
		self.y = self.y[:self.is_train]

		print "x_train =", self.x.shape
		print "y_train =", len(self.y)#.shape
		print "x_valid =", self.x_valid.shape
		print "y_valid =", len(self.y_valid)#.shape'''

	def set_container(self):
		self.lw = np.zeros([self.batch_size, 1, self.vocab_size], dtype = int)
		self.lw[:, 0, self.vocab_size - 2] = 1
		self.lw2 = np.zeros([self.batch_size, 1, self.vocab_size], dtype = int)
		self.lw2[:, 0, self.vocab_size - 1] = 1
		self.mask_zero = np.zeros([self.batch_size, self.num_steps_out])
		#self.h_time_zero = np.empty([self.batch_size, self.hidden1_size * self.h_time])

		self.sampling_id = [0.01, 0.02, 0.5, 0.8, 1]
		self.sampling = 1
		
	'''def cut_batch(self, step, is_train, is_valid):
		self.batch_id = step % self.batch_num
		begin = self.batch_size * self.batch_id
		end = self.batch_size * (self.batch_id + 1)
		
		if is_valid == False:
			self.x_batch = self.x[begin:end]
			temp = self.y[begin:end]
		else:
			self.x_batch = self.x_valid[begin:end]
			temp = self.y_valid[begin:end]

		self.y_batch = []
		for i in range(0, self.batch_size): 
			if is_train == 1: 
				y_idx = random.randint(0, len(temp[i])-1)
				#y_idx = 0 # !!!
			else: y_idx = 0
			self.y_batch = self.y_batch + [temp[i][y_idx]]
		self.y_batch = np.array(self.y_batch)

		count = 0.0
		self.mask = self.mask_zero[:]
		for i in range(self.batch_size):
			j = np.where(self.y_batch[i] == self.EOS_id)[0][0]
			count += j
			self.mask[i][:j] = 1

		self.count = count

		self.y_batch = self.hoty(self.y_batch)
		#print "batch", x_batch.shape, y_batch.shape
		if self.sampling == 1 or True:
			#self.lw_batch = np.concatenate([self.lw, self.y_batch[:, 0:self.num_steps_out-1]], axis = 1)
			self.lw_batch = self.y_batch[:]
		#print lw_batch.shape # 50, 44, 5995
		self.y_batch = np.concatenate([self.y_batch[:, 1:self.num_steps_out], self.lw2], axis = 1)
		return'''
		
	def hotx(self, arr):
		new_arr = np.zeros((self.batch_size, self.num_steps, self.dim_size), dtype = int)
		for i in range(self.batch_size):
			new_arr[i][np.arange(arr.shape[1]), arr[i]] = 1
		return new_arr

	def hoty(self, arr):
		new_arr = np.zeros((self.batch_size, self.num_steps_out, self.vocab_size), dtype = int)
		for i in range(self.batch_size):
			try:
				new_arr[i][np.arange(arr.shape[1]), arr[i]] = 1
			except IndexError:
				print "IndexError!", arr[i]
				fout.write("IndexError!" + arr[i] + "\n")
		return new_arr

	def set_nn_parameter(self):
  		print "Setting parameter"
  		self.win1 = tf.Variable(tf.random_normal([self.dim_size, self.hidden1_size]))
		self.bin1 = tf.Variable(tf.constant(0.1, shape = [self.hidden1_size, ]))
		self.win2 = tf.Variable(tf.random_normal([self.hidden1_size, self.hidden2_size]))
		self.bin2 = tf.Variable(tf.constant(0.1, shape = [self.hidden2_size, ]))

		#self.wh1 = tf.Variable(tf.random_normal([self.num_steps_out, self.h_time]))
		self.wh1 = tf.Variable(tf.random_normal([self.hidden1_size, self.w_size])) # h*w*lw
		#self.bh1 = tf.Variable(tf.constant(0.1, shape = [self.hidden2_size, ])) # for h1
		self.wh2 = tf.Variable(tf.random_normal([self.vocab_size, self.w_size])) # for last word
		self.bh2 = tf.Variable(tf.constant(0.1, shape = [self.w_size, ]))


		self.out1_size = self.hidden1_size + self.w_size
		#self.out1_size = self.hidden1_size + self.vocab_size
		self.wout1 = tf.Variable(tf.random_normal([self.out1_size, self.hidden2_size]))
		self.bout1 = tf.Variable(tf.constant(0.1, shape = [self.hidden2_size, ]))
		self.wout2 = tf.Variable(tf.random_normal([self.hidden2_size, self.vocab_size]))
		self.bout2 = tf.Variable(tf.constant(0.1, shape = [self.vocab_size, ]))

		self.x_nn = tf.placeholder(tf.float32, [None, self.num_steps, self.dim_size])
		self.y_nn = tf.placeholder(tf.float32, [None, self.num_steps_out, self.vocab_size])
		
		self.all_last_word = tf.placeholder(tf.float32, [None, self.num_steps_out, self.vocab_size])
		self.mask_nn = tf.placeholder(tf.float32, [None, self.num_steps_out])
		
		self.keep_prob = tf.placeholder("float")
		self.bs = tf.placeholder(dtype = tf.int32)
		self.acc_count = tf.placeholder("float")
		
	def select_time(self, h):
		#temp = h[:, self.n-1:self.n, :]
		#for i in range(1, self.h_time): 
		#	temp = tf.concat([temp, h[:, i*self.n+self.n-1:(i+1)*self.n, :]], 1)
		time_list = [10, 20, 30, 40, 50, 60, 65, 70, 75, 79]	
		#time_list = [79]
		temp = h[:, time_list[0]:time_list[0]+1]
		for i in range(1, self.h_time):
			temp = tf.concat([temp, h[:, time_list[i]:time_list[i]+1]], 1)
		#temp = h[:, time_list, :]
		#temp = tf.reshape(temp, (-1, self.h_time, self.hidden2_size))
		return temp

	def hidden(self, h, time, lw):
		with tf.variable_scope("hidden_layer"):
		#for xx in range(1): #print xx, "---"
			if True:
				#lw = tf.nn.relu(tf.matmul(self.all_last_word[:, time], self.wh2) + self.bh2)
				lw = (tf.matmul(self.all_last_word[:, time], self.wh2) + self.bh2)
				#lw = self.all_last_word[:, time]

			h1 = []
			for i in range(self.batch_size):
				temp = tf.matmul(h[i], self.wh1)
				temp2 = tf.reshape(lw[i], (self.w_size, 1))
				alpha = tf.matmul(temp, temp2)
				#alpha = tf.nn.softmax(temp) # shape = 10*1 for each batch
				idx = tf.to_int32(tf.argmax(alpha, 0)[0])
				#idx = 9
				h1.append(h[i][idx])
			h2 = tf.concat([tf.stack(h1), lw], 1)
		#exit()
		return h2

	def hidden_test(self, h, time, lw):
		with tf.variable_scope("hidden_layer"):
		#for xx in range(1): #print xx, "---"
			
			if True:
				if time != 0:
					#print "nonono"
					#lw = tf.nn.softmax(lw, -1)
					lw = tf.one_hot(tf.argmax(lw, 1), depth = self.vocab_size)
				lw = tf.cast(lw, tf.float32)
				lw = (tf.matmul(lw, self.wh2) + self.bh2)

			h1 = []
			for i in range(self.batch_size):
				#temp = tf.matmul(h[i], self.wh1)
				#temp2 = tf.reshape(lw[i], (self.w_size, 1))
				#alpha = tf.matmul(temp, temp2)
				#alpha = tf.nn.softmax(temp) # shape = 10*1 for each batch
				#idx = tf.to_int32(tf.argmax(alpha, 0)[0])
				idx = 9
				h1.append(h[i][idx])
			h2 = tf.concat([tf.stack(h1), lw], 1)
		#exit()
		return h2

	def lstm_in_layer(self, X): # dropout set at in/ out of rnn!!!
		X = tf.reshape(X, [-1, self.dim_size])
		#X_in = (tf.matmul(X, self.win1) + self.bin1)
		with tf.variable_scope("RNN_in"):
			X_in = tf.nn.relu(tf.matmul(X, self.win1) + self.bin1)
			X_in = tf.reshape(X_in, [-1, self.num_steps, self.hidden1_size])
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden1_size, forget_bias = 1.0, state_is_tuple = True)
			init_state = lstm_cell.zero_state(self.bs, dtype = tf.float32)
			outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = init_state, time_major = False)
			#outputs = tf.reshape(outputs, [-1, self.hidden1_size])
			#outputs = tf.nn.relu(tf.matmul(outputs, self.win2) + self.bin2)
			outputs = tf.reshape(outputs, [-1, self.num_steps, self.hidden1_size])
		return outputs

	def lstm_in_layer_test(self, X): # dropout set at in/ out of rnn!!!
		X = tf.reshape(X, [-1, self.dim_size])
		#X_in = (tf.matmul(X, self.win1) + self.bin1)
		with tf.variable_scope("RNN_in"):
			tf.get_variable_scope().reuse_variables()
			X_in = tf.nn.relu(tf.matmul(X, self.win1) + self.bin1)
			X_in = tf.reshape(X_in, [-1, self.num_steps, self.hidden1_size])
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden1_size, forget_bias = 1.0, state_is_tuple = True)
			init_state = lstm_cell.zero_state(self.bs, dtype = tf.float32)
			outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = init_state, time_major = False)
			#outputs = tf.reshape(outputs, [-1, self.hidden1_size])
			#outputs = tf.nn.relu(tf.matmul(outputs, self.win2) + self.bin2)
			outputs = tf.reshape(outputs, [-1, self.num_steps, self.hidden1_size])
		return outputs
	
	def lstm_out_layer(self, h):
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden2_size, forget_bias = 1.0, state_is_tuple = True)
		state = lstm_cell.zero_state(self.bs, dtype = tf.float32)
		outputs = self.lw[:, 0, :]
		h = self.select_time(h) # h[:, -1, :]
		#print "H2"
		with tf.variable_scope("RNN"):
			for time in range(self.num_steps_out):
				#print "1"
				X = self.hidden(h, time, outputs) # combine hidden CNN and last word
				#print "2"
				X_in = tf.nn.relu(tf.matmul(X, self.wout1) + self.bout1) # relu!
				if time > 0: tf.get_variable_scope().reuse_variables()
				outputs, state = lstm_cell(X_in, state) # X_in[:, :]
				outputs = tf.nn.relu(tf.matmul(outputs, self.wout2) + self.bout2)
				#outputs = tf.nn.relu(tf.matmul(outputs, self.wout2) + self.bout2)
				
				if time != 0: results = tf.concat([results, outputs], 0)
				else: results = outputs
				#if time == 10: exit()
		
		results = tf.reshape(results, (self.num_steps_out, -1, self.vocab_size))
		results = tf.transpose(results, perm = [1, 0, 2])
		return results

	def lstm_out_layer_test(self, h):
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden2_size, forget_bias = 1.0, state_is_tuple = True)
		state = lstm_cell.zero_state(self.bs, dtype = tf.float32)
		outputs = self.lw[:, 0, :]
		h = self.select_time(h) # h[:, -1, :]
		#print "H2"
		with tf.variable_scope("RNN"):
			for time in range(self.num_steps_out):
				#print "1"
				X = self.hidden(h, time, outputs) # combine hidden CNN and last word
				#print "2"
				X_in = tf.nn.relu(tf.matmul(X, self.wout1) + self.bout1) # relu!
				if time > 0: tf.get_variable_scope().reuse_variables()
				outputs, state = lstm_cell(X_in, state) # X_in[:, :]
				outputs = tf.nn.relu(tf.matmul(outputs, self.wout2) + self.bout2)
				#outputs = tf.nn.relu(tf.matmul(outputs, self.wout2) + self.bout2)
				
				if time != 0: results = tf.concat([results, outputs], 0)
				else: results = outputs
				#if time == 10: exit()
		
		results = tf.reshape(results, (self.num_steps_out, -1, self.vocab_size))
		results = tf.transpose(results, perm = [1, 0, 2])
		return results

	def lstm_out_layer_test(self, h):
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden2_size, forget_bias = 1.0, state_is_tuple = True)
		state = lstm_cell.zero_state(self.bs, dtype = tf.float32)
		outputs = self.lw[:, 0, :]
		h = self.select_time(h) # h[:, -1, :]
		#print "H2"
		with tf.variable_scope("RNN"):
			for time in range(self.num_steps_out):
				#print "1"
				X = self.hidden_test(h, time, outputs) # combine hidden CNN and last word
				#print "2"
				X_in = tf.nn.relu(tf.matmul(X, self.wout1) + self.bout1) # relu!
				if time > 0 or True: tf.get_variable_scope().reuse_variables()
				outputs, state = lstm_cell(X_in, state) # X_in[:, :]
				outputs = tf.nn.relu(tf.matmul(outputs, self.wout2) + self.bout2)
				#outputs = tf.nn.relu(tf.matmul(outputs, self.wout2) + self.bout2)
				
				if time != 0: results = tf.concat([results, outputs], 0)
				else: results = outputs
				#if time == 10: exit()
		
		results = tf.reshape(results, (self.num_steps_out, -1, self.vocab_size))
		results = tf.transpose(results, perm = [1, 0, 2])
		return results

	'''def compute_sampling_rate(self, step):
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
			return 2'''

	'''def summary(self):
		print "Sampling = ", 
		for i in range(len(self.sampling_id)):
			print (self.sampling_id[i] * self.epoch),
		print ""
		print "Total epoch =", self.epoch
		print "LSTM units =", self.hidden1_size, self.hidden2_size
		print "H-time =", self.h_time
		self.fout.write("\nTotal epoch = " + str(self.epoch) + " ")
		self.fout.write("LSTM units = " + str(self.hidden1_size) + " + " + str(self.hidden2_size) + " ")
		self.fout.write("H-time = " + str(self.h_time) + "\n\n")'''

	def output_accuracy(self):
		print "%4d %2d %1d" %(self.step, self.batch_id, self.sampling), 
		self.fout.write("%4d %2d %1d, " %(self.step, self.batch_id, self.sampling))
		acc2 = 0
		loss2 = 0
		n = 7
		for i in range(self.batch_num/n):
			self.cut_batch(n*i, 0, False)

			acc2 += self.sess.run(self.accuracy, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
				self.bs: self.batch_size, self.keep_prob: self.drop,
				self.acc_count: self.count})
			
			loss2 += self.sess.run(self.loss, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
				self.bs: self.batch_size, self.keep_prob: self.drop,
				self.acc_count: self.count})
			
		acc2 /= (self.batch_num/n)
		loss2 /= (self.batch_num/n)
	
		print "| acc: %.3f" % acc2,
		print "loss: %.3f" % (loss2 / self.batch_size * 100),
		self.fout.write("| acc: %.3f, " % acc2)
		self.fout.write("loss: %.3f, " % (loss2 / self.batch_size * 100))
		
		m = 1
		if self.valid_num >= m:
			v_acc = 0
			v_loss = 0
			v_loss2 = 0
			self.sampling = 2
			for i in range(self.valid_num):	
				self.cut_batch(i, 0, True)

				v_acc += self.sess.run(self.accuracy, feed_dict = {
					self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
					self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
					self.bs: self.batch_size, self.keep_prob: self.drop,
					self.acc_count: self.count})
				v_loss += self.sess.run(self.loss, feed_dict = {
					self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
					self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
					self.bs: self.batch_size, self.keep_prob: self.drop,
					self.acc_count: self.count})

				v_loss2 += self.sess.run(self.loss_test, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
				self.bs: self.batch_size, self.keep_prob: self.drop,
				self.acc_count: self.count})
				#v_acc += sess.run(accuracy, feed_dict = {x_nn: x_valid, y_nn: y_valid,}) 
			v_acc /= (self.valid_num)
			v_loss /= (self.valid_size)
			v_loss2 /= (self.valid_size)
			
			print "| v_acc: %.3f" % v_acc,
			print "v_loss: %.3f" % (v_loss * 100), 
			print "| %.3f" % (v_loss2 * 100)
			self.fout.write("| acc: %.3f, " % v_acc)
			self.fout.write("loss: %.3f, " % (v_loss * 100))
			self.fout.write("%.3f\n" % (v_loss2 * 100))
		
	def nn_model(self):
		print "Constructing rnn model"
		h = self.lstm_in_layer(self.x_nn)
		print "H1"
		self.pred = self.lstm_out_layer(h)
		print "H3"
		loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_nn, logits = self.pred)
		#+ tf.nn.l2_loss(self.w1)*self.beta1 + tf.nn.l2_loss(self.w4)*self.beta2)
		#print self.loss.shape # batch, 44
		#print self.y_nn.shape # 50, 44, 5995
		#print self.mask.shape # 50, 44
		loss = loss * self.mask_nn
		#print self.loss.shape # 50, 44
		self.loss = tf.reduce_mean(loss)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_step = optimizer.apply_gradients(zip(grads, tvars))
		correct_pred = tf.equal(tf.argmax(self.pred, 2), tf.argmax(self.y_nn, 2))
		#self.t = correct_pred
		#print correct_pred.shape # batch * 
		correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32)) - (tf.to_float(self.bs)*tf.to_float(self.num_steps_out) - self.acc_count)
		self.accuracy = correct/(self.acc_count)
		#self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		#self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		#init = tf.global_variables_initializer()

	def nn_test_model(self):
		print "Constructing rnn testing model"
		h = self.lstm_in_layer_test(self.x_nn)
		self.pred_test = self.lstm_out_layer_test(h)

		loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_nn, logits = self.pred_test)
		loss = loss * self.mask_nn
		self.loss_test = tf.reduce_mean(loss)
		self.sess = tf.Session()
		
		


	def one_epoch(self):
		self.cut_batch(self.step, is_train = True, is_valid = False)

		self.sess.run([self.train_step], feed_dict = {
			self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
			self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
			self.bs: self.batch_size, self.keep_prob: self.drop, self.acc_count: self.count})
		
	'''def training(self):
		self.sess = tf.Session()
		self.step = 0

		if self.first_epoch == 0:
			
	  	else: self.sess.run(self.init)
		print "\nStart training"

		while self.step < self.epoch:
			self.sampling = 1
			#self.sampling = self.compute_sampling_rate(self.step) # 1 for true, 2 for mine
			self.one_epoch()
		
			if self.step % 101 == 0: self.output_accuracy()
			
			if self.step == 2900 or self.step == 5800:
				self.lr *= 0.3
				print "lr = ", self.lr
			self.step += 1
		
		self.output_accuracy()		
		save_path = self.saver.save(self.sess, "../model/model" + self.name + ".ckpt")
		print("Model saved in file: %s" % save_path)'''

	def load_testing(self):
		self.saver.restore(self.sess, "./hw2/model/model" + self.name + ".ckpt")
	 	print("Model restored.")
		self.id_word = pickle.load(open("id_word.jpg", "rb"))
		self.x_test = np.load("feat_test.npy")
		print self.x_test
		
		#self.sampling = 2
		self.fout_test = open('src/sentence_test_all','w')


	'''def training_output(self):
		self.sampling = 2
		self.step = 0
		n = 0
		fout = open('../testing_src/sentence_training3','w')

		for self.sampling in [1, 2, 2]:
			#self.one_epoch()

			#self.sampling = self.step%2 + 1
			#if self.step == 3: self.step = 28
			self.cut_batch(self.step, 0, True)

			if n == 2:
				self.lw_batch = (np.zeros([self.batch_size, self.num_steps_out, self.vocab_size], dtype = int))
				self.lw_batch[:, 0, self.vocab_size - 2] = 1
			
			if n == 0:
				pred = self.sess.run(self.pred, feed_dict = {
					self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
					self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
					self.bs: self.batch_size, self.keep_prob: self.drop,
					self.acc_count: self.count})

			else:
				pred = self.sess.run(self.pred_test, feed_dict = {
					self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
					self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
					self.bs: self.batch_size, self.keep_prob: self.drop,
					self.acc_count: self.count})

			print self.sampling
			n += 1
			pred = self.sess.run(tf.nn.softmax(pred, -1))
			pred = self.sess.run(tf.argmax(pred, 2))
			print pred[0][0]
			for i in range(5):
				i0 = self.step*50 + i
				#print i0
				fout.write(str(i) + " " + str(self.step) + " " + str(self.sampling) + " ")
				for j in range(self.num_steps_out - 30):
					fout.write(self.id_word[self.y_valid[i0][0][j]] + " ")
				fout.write("\n")
				for j in range(self.num_steps_out - 20):
					fout.write(self.id_word[pred[i][j]] + " ")
				fout.write("\n")

			#self.step += 1
		fout.close()'''

	def cut_batch_test(self, step):
		self.x_batch = self.x_test[step*50:(step+1)*50]
		
		#self.lw_batch = 
		return
		
	def testing(self):
		#self.sampling = 2
		self.lw_batch = (np.zeros([self.batch_size, self.num_steps_out, self.vocab_size], dtype = int))
		self.lw_batch[:, 0, self.vocab_size - 2] = 1
		for step in range(2):
			if self.x_test.shape[0] == step*50: break
			self.cut_batch_test(step)
			pred = self.sess.run(self.pred_test, feed_dict = {
				self.x_nn: self.x_batch, 
				self.all_last_word: self.lw_batch,
				self.bs: self.batch_size, self.keep_prob: self.drop,})

			#print self.sampling, "lala"	
			pred = self.sess.run(tf.nn.softmax(pred, -1))	
			pred = self.sess.run(tf.argmax(pred, 2))
				
			#print pred[0]
			#print pred[10]

			self.output(pred)
		#return pred
			
	def output(self, pred):
		for i in range(50):
			for j in range(self.num_steps_out - 20):
				self.fout_test.write(self.id_word[pred[i][j]] + " ")
			self.fout_test.write("\n")

	def ans_all(self):
		ans = []
		for i in range(50): ans.append([])

		for i in range(10):
			for j in range(5): 
				self.one_epoch()
			pred = self.testing()
			'''for j in range(50):
				t = []
				for k in range(self.num_steps_out - 20):
					t.append(self.id_word[pred[j][k]])
				ans[j].append(t)'''
		
		'''for i in range(50):
			for j in range(10):
				for k in range(len(ans[i][j])):
					self.fout_test.write(ans[i][j][k] + " ")
				self.fout_test.write("\n")'''

				
	 
model = Model()
#model.load()
model.set_nn_parameter()
model.set_container()
model.nn_model()
model.nn_test_model()
model.init = tf.global_variables_initializer()
#model.summary()
#model.training()

model.load_testing()
#model.training_output()

model.testing()
#model.ans_all()
print "\n Compile successfully!"

