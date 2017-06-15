import tensorflow as tf
import numpy as np
import pickle
import sys
import random
import seq2seq_lib
import math

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config = config))

class Model(object):
	def __init__(self, path):
		print "\nSeq2Seq + Reinforcement Learning for Chatbot, 06/2017\n"
		self.first_train = 0
		self.drop = 0.8
		self.lr = 3e-4 # 0.5 -> *= 0.99
		self.lr_RL = 10e-4 # 0.5 -> *= 0.99

		# dimension
		self.num_steps = 15
		self.num_layers = 2
		self.vocab_size = 10000
		self.encoder_size = 512
		self.decoder_size = 512

		# constant
		self.PAD = 0
		self.BOS = 1
		self.EOS = 2
		self.Unknown = 3

		self.lam1 = 0.15 # dull set
		self.lam2 = 0.05 # EOS, Unknown
		self.lam3 = 0.85
		#self.b1 = tf.constant(10.0) # baseline: -log(12/128) = 2.37
		#self.b2 = tf.constant(40.0)
		#self.b3 = tf.constant(1.0)
		self.b1 = 10.0 # > 0
		self.b2 = 40.0 # > 0
		self.b3 = -3.0 # < 0

		# number of data
		self.data_size = 158979
		self.train_size = 1700*128
		self.batch_size = 64
		self.train_batch = self.train_size / self.batch_size # = 3400
		self.valid_size = self.data_size - self.train_size # num of valid
		self.valid_batch = self.valid_size / self.batch_size
		self.total_epoch = self.train_batch*10/34  #self.train_batch * 30) 
		
		self.path = path
		
		self.sess = tf.Session()
		
	def load(self):
		#self.x, self.y = seq2seq_lib.load("corpus_15.npy", self.train_size)
		self.id_word = pickle.load(open("./id_word_dict", "rb"))
		#self.dull = np.load("./dull.npy")
		#print "dull =", self.dull.shape
		#print self.dull

		#self.name = str(self.encoder_size) + "-" + str(self.decoder_size)
		self.name = "-512-RL"
		self.fout = open('record' + self.name, 'w')
		self.fout_tt = open(sys.argv[3], 'w')
		self.fout_test = open('testing_results' + self.name,'w')
		self.fout_results = open('training_results' + self.name,'w')

	def set_container(self):
		self.BOS_token = np.zeros([self.batch_size, self.vocab_size], dtype = int)
		self.BOS_token[:, 1] = 1
		self.mask_zero = np.zeros([self.batch_size, self.num_steps-1])
		self.x_batch = np.zeros([self.batch_size, self.num_steps, self.vocab_size])

		self.sampling_id = [0.1, 0.3, 0.5, 0.7, 0.9]
		self.sampling = 1
		
	def set_variable(self):
  		print "Setting parameter"
  		self.w_en1 = tf.Variable(tf.random_normal([self.vocab_size, self.encoder_size]))
		self.b_en1 = tf.Variable(tf.constant(0.1, shape = [self.encoder_size, ]))

		#self.w_h1 = tf.Variable(tf.random_normal([self.num_steps, self.h_time]))
		self.w_h1 = tf.Variable(tf.random_normal([self.vocab_size, self.decoder_size])) # h*w*lw
		self.b_h1 = tf.Variable(tf.constant(0.1, shape = [self.decoder_size, ])) # for h1

		self.hidden_size = self.encoder_size + self.decoder_size
		#self.out1_size = self.hidden1_size + self.vocab_size
		self.w_de1 = tf.Variable(tf.random_normal([self.hidden_size, self.decoder_size]))
		self.b_de1 = tf.Variable(tf.constant(0.1, shape = [self.decoder_size, ]))
		self.w_de2 = tf.Variable(tf.random_normal([self.decoder_size, self.vocab_size]))
		self.b_de2 = tf.Variable(tf.constant(0.1, shape = [self.vocab_size, ]))

		self.x_nn = tf.placeholder(tf.float32, [None, self.num_steps, self.vocab_size])
		self.y_nn = tf.placeholder(tf.float32, [None, self.num_steps, self.vocab_size])
		self.pred_nn = tf.placeholder(tf.float32, [None, self.num_steps])

		#self.all_last_word = tf.placeholder(tf.float32, [None, self.num_steps_out, self.vocab_size])
		self.mask_nn = tf.placeholder(tf.float32, [None, self.num_steps-1])
		
		self.bs = tf.placeholder(dtype = tf.int32)
		self.keep_prob = tf.placeholder("float")
		self.acc_count = tf.placeholder("float")
		#self.test = tf.placeholder_with_default(False, shape = [1])
		#self.test = False

	def cut_batch(self, step, is_train = True, sampling = 1):
		n = step % self.train_batch
		begin = self.batch_size * n
		end = self.batch_size * (n + 1)
		
		if is_train == True:
			self.x_batch = self.x[begin:end]
			self.y_batch = self.y[begin:end]
		else:
			self.x_batch = self.x_valid[begin:end]
			self.y_batch = self.y_valid[begin:end]

		count = 0.0 # number of total words need to be computed loss in one batch, 
		self.mask = self.mask_zero[:]
		for i in range(self.batch_size):
			j = np.where(self.y_batch[i] == self.EOS)[0][0]		
			count += j+1
			self.mask[i][:j+1] = 1	
		self.count_words = count
		
		self.x_batch = seq2seq_lib.one_hot(\
			self.x_batch, self.batch_size, self.num_steps, self.vocab_size)
		self.y_batch = seq2seq_lib.one_hot(\
			self.y_batch, self.batch_size, self.num_steps, self.vocab_size)
		
		return

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

	def rnn_encoder(self, X): # dropout set at in/ out of rnn!!!
		X = tf.reshape(X, [-1, self.vocab_size])

		with tf.variable_scope("RNN_Encoder"):
			X = (tf.matmul(X, self.w_en1) + self.b_en1)
			X = tf.reshape(X, [-1, self.num_steps, self.encoder_size])
			cell = tf.contrib.rnn.BasicLSTMCell(self.encoder_size, forget_bias = 1.0, 
				state_is_tuple = True)
			init_state = cell.zero_state(self.bs, dtype = tf.float32)
			outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state = init_state, 
				time_major = False)
			
			'''single_cell = tf.nn.rnn_cell.GRUCell(size)
			cell = single_cell
        	if num_layers > 1: cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)'''

			outputs = tf.reshape(outputs, [-1, self.num_steps, self.encoder_size])
		return outputs
	
	def rnn_decoder(self, h, y_nn, test = False): # for input the real last word into decoder layer
		cell = tf.contrib.rnn.BasicLSTMCell(self.decoder_size, forget_bias = 1.0, 
			state_is_tuple = True)
		if not test:  # TODO: Should use a placeholder instead
			cell = tf.contrib.rnn.DropoutWrapper(cell,
				input_keep_prob = 1.0, output_keep_prob = self.drop)
		state = cell.zero_state(self.bs, dtype = tf.float32)
		if test == True: last_word = tf.cast(self.BOS_token, tf.float32)
		h_atten = h[:, -1, :]

		with tf.variable_scope("RNN_Decoder"):
			for time in range(self.num_steps):
				if test == False: last_word = y_nn[:, time, :]
				if test == True or time > 0: tf.get_variable_scope().reuse_variables()
				
				#X = self.hidden(h, time, outputs) # combine code representation and last word
				#last_word = tf.nn.relu(tf.matmul(last_word, self.w_h1) + self.b_h1)
				last_word = (tf.matmul(last_word, self.w_en1) + self.b_en1)

				X = tf.concat([h_atten, last_word], 1)
				X = tf.nn.relu(tf.matmul(X, self.w_de1) + self.b_de1) # relu!
				outputs, state = cell(X, state)
				outputs = tf.nn.relu(tf.matmul(outputs, self.w_de2) + self.b_de2)
				
				if test == True:
					last_word = tf.argmax(outputs, 1)
					last_word = tf.one_hot(last_word, self.vocab_size)
					#last_word = tf.cast(last_word, tf.int32)
				
				if time != 0: results = tf.concat([results, outputs], 0)
				else: results = outputs
		
		results = tf.reshape(results, (self.num_steps, -1, self.vocab_size))
		results = tf.transpose(results, perm = [1, 0, 2])
		print "pred =", results.get_shape()
		return results

	def summary(self):
		#print "Sampling = ", 
		#for i in range(len(self.sampling_id)): print (self.sampling_id[i] * self.epoch),
		print ""
		print "Total Epoch =", self.total_epoch
		print "RNN units =", self.encoder_size, self.decoder_size
		#print "H-time =", self.h_time
		self.fout.write("\nTotal Epoch = " + str(self.total_epoch) + " ")
		self.fout.write("RNN units = " + str(self.encoder_size) + " + " + str(self.decoder_size) + " ")
		#self.fout.write("H-time = " + str(self.h_time) + "\n\n")

	def output_accuracy(self):
		print "%4d %2d %1d" %(self.step, self.step/self.train_batch, self.sampling), 
		self.fout.write("%4d %2d, " %(self.step, self.step/self.train_batch))
		acc_t = 0
		loss_t = 0
		sample_num = 100
		for i in range(self.train_batch/sample_num):
			self.cut_batch(sample_num*i)

			acc_t += self.sess.run(self.accuracy, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.mask_nn: self.mask, self.acc_count: self.count_words,
				self.bs: self.batch_size, self.keep_prob: self.drop})
			
			loss_t += self.sess.run(self.loss, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.mask_nn: self.mask, self.acc_count: self.count_words,
				self.bs: self.batch_size, self.keep_prob: self.drop})
			
		acc_t /= (self.train_batch/sample_num)
		loss_t /= (self.train_batch/sample_num)
	
		print "| acc: %.3f" % (acc_t*100),
		print "loss: %.3f" % (loss_t / self.batch_size * 1000)
		self.fout.write("| acc: %.3f, " % (acc_t*100))
		self.fout.write("loss: %.3f, \n" % (loss_t / self.batch_size * 1000))
		
		'''m = 1
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
			self.fout.write("%.3f\n" % (v_loss2 * 100))'''
		
	def output_reward(self):
		print "%4d %2d" %(self.step, self.step/self.train_batch), 
		self.fout.write("%4d %2d, " %(self.step, self.step/self.train_batch))
		r1_t = 0
		r2_t = 0
		r3_t = 0
		sample_num = 200
		for i in range(self.train_batch/sample_num):
			self.cut_batch(sample_num*i)

			r1_t += self.sess.run(self.r1, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.bs: self.batch_size, self.keep_prob: self.drop})
			r2_t += self.sess.run(self.r2, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.bs: self.batch_size, self.keep_prob: self.drop})
			r3_t += self.sess.run(self.r3, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.bs: self.batch_size, self.keep_prob: self.drop})
			
		r1_t /= (self.train_batch/float(sample_num))
		r2_t /= (self.train_batch/float(sample_num))
		r3_t /= (self.train_batch/float(sample_num))

		R = self.lam1*(r1_t-self.b1) + self.lam2*(r2_t-self.b2) + self.lam3*(r3_t-self.b3)
	
		print "| r1: %.3f" % (r1_t), "| r2: %.3f" % (r2_t), "| r3 : %.3f" % (r3_t), "| R : %.3f" % (R)
		self.fout.write("| r1: %.3f" % (r1_t) + "| r2: %.3f" % (r2_t) + "| r3: %.3f" % (r3_t) + "| R: %.3f, \n" % (R))

	def build_model(self):
		print "Constructing RNN Model..."
		hidden = self.rnn_encoder(self.x_nn)
		self.pred = self.rnn_decoder(hidden, self.y_nn, test = False)
		self.pred_test = self.rnn_decoder(hidden, self.y_nn, test = True)

		loss = tf.nn.softmax_cross_entropy_with_logits(
			labels = self.y_nn[:, 1:, :], logits = self.pred[:, :self.num_steps-1, :])
		loss = loss * self.mask_nn
		self.loss = tf.reduce_mean(loss)

		with tf.variable_scope("train"):
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
			optimizer = tf.train.AdamOptimizer(self.lr)
			self.train_step = optimizer.apply_gradients(zip(grads, tvars))

		correct_pred = tf.equal(tf.argmax(self.pred[:, :self.num_steps-1, :], 2), tf.argmax(self.y_nn[:, 1:, :], 2))
		correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32)) - (
			tf.to_float(self.bs)*tf.to_float(self.num_steps-1) - self.acc_count)
		self.accuracy = correct/(self.acc_count)

		# for RL
		'''self.r1 = self.logProb1(self.pred_test) 
		self.r2 = self.logProb2(self.pred_test) 
		self.r3 = self.logProb3(self.pred_test, self.y_nn)
		R1 = tf.multiply(self.lam1, tf.subtract(self.r1, self.b1))
		R2 = tf.multiply(self.lam2, tf.subtract(self.r2, self.b2))
		R3 = tf.multiply(self.lam3, tf.subtract(self.r3, self.b3))
		self.R = R1 + R2 + R3
		
		with tf.variable_scope("train_RL"):
			tvars = tf.trainable_variables()
			self.train_step_RL = tf.train.GradientDescentOptimizer(self.lr_RL).minimize(- self.R, var_list = tvars)
			self.train_step_RL3 = tf.train.GradientDescentOptimizer(self.lr_RL).minimize(- R3, var_list = tvars)
			#self.train_step_RL = tf.train.AdamOptimizer(self.lr_RL).minimize(-self.R, var_list = tvars)
			#self.train_step_RL3 = tf.train.AdamOptimizer(self.lr_RL).minimize(-R3, var_list = tvars)'''
		
		self.saver = tf.train.Saver()

	

	def logProb1(self, pred):
		print "Building R1..."
		sum_p = []
		#y_nn = tf.argmax(y_nn, -1)
		for i in range(self.dull.shape[0]):
			for j in range(self.batch_size): # remove redundant calculation
				total_p = 0 
				n = np.where(self.dull[i] == 2)[0][0] - 1 # there are n real words
				for k in range(n): 
					p = tf.nn.softmax(pred[j][k], -1)[tf.cast(self.dull[i][k+1], tf.int32)]
					p = tf.cond(p < 1e-30, lambda: tf.constant(1e-30), lambda: p)
					total_p += tf.log(p)
					#print total_p
	        	sum_p.append(total_p/float(n))
       
		return -np.sum(sum_p)/len(sum_p)

	def logProb2(self, pred):
		print "Building R2..."
		sum_p = []
		for token in [self.EOS, 0]:
			for i in range(self.batch_size): # remove redundant calculation
				total_p = 0 
				#n = tf.where(y_nn[i] == 2) - 1 # there are n real words
				for k in range(self.num_steps - 1): 
					p = tf.nn.softmax(pred[i][k], -1)[tf.cast(token, tf.int32)]
					p = tf.cond(p < 1e-30, lambda: tf.constant(1e-30), lambda: p)
					total_p += tf.log(p)
					#print total_p
	        	sum_p.append(total_p/float(self.num_steps - 1))
       
		return -np.sum(sum_p)/len(sum_p)

	def logProb3(self, pred, y_nn):
		print "Building R3..."
		sum_p = []
		y_nn = tf.argmax(y_nn, -1)
		for i in range(self.batch_size): # remove redundant calculation
			total_p = 0 
			#n = tf.where(y_nn[i] == 2) - 1 # there are n real words
			for k in range(self.num_steps - 1): 
				p = tf.nn.softmax(pred[i][k], -1)[tf.cast(y_nn[i][k+1], tf.int32)]
				p = tf.cond(p < 1e-30, lambda: tf.constant(1e-30), lambda: p)
				total_p += tf.log(p)
				#print total_p
        	sum_p.append(total_p/float(self.num_steps - 1))
       
		return np.sum(sum_p)/len(sum_p)

	def one_step(self):
		if self.sampling == 1:
			self.sess.run(self.train_step, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.mask_nn: self.mask, self.acc_count: self.count_words, 
				self.bs: self.batch_size, self.keep_prob: self.drop})
		elif self.sampling == 2:
			self.sess.run([self.train_step_2], feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.all_last_word: self.lw_batch, self.mask_nn: self.mask,
				self.bs: self.batch_size, self.keep_prob: self.drop, self.acc_count: self.count})
		#print "Finished training for 1 batch"

	def one_step_RL(self, rewardID = None):
		if rewardID == 3:
			self.sess.run(self.train_step_RL3, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.bs: self.batch_size, self.keep_prob: self.drop})
		else:
			self.sess.run(self.train_step_RL, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.bs: self.batch_size, self.keep_prob: self.drop})
		
	def initialize_model(self):
		self.init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.step = 0

		self.sess.run(self.init)
		name = "-512-RL"
		if self.first_train == 0:
			self.saver.restore(self.sess, self.path+".ckpt")
	 		print("Model restored.")

	def training(self):
		print "\nStart training"
		for self.step in range (self.total_epoch):
			self.cut_batch(self.step)

			#if self.step % 20 == 0: self.one_step()
			#if self.step % 20 == 0: 
			self.one_step_RL()
			#else: 
			#self.one_step_RL(rewardID = 3)

			if self.step % 100 == 0 : 
				self.output_accuracy()
				self.output_reward()	
				self.testing()

			if self.step % 2000 == 0 and self.step != 0: 
				save_path = self.saver.save(self.sess, "../model/model" + self.name + ".ckpt")
				print("Model saved in file: %s" % save_path)

			#if self.step % 10000 == 0 and self.step != 0:
				#	self.lr *= 0.3
				#	print "lr = ", self.lr
		
		self.output_accuracy()	
		self.output_reward()
		self.testing()
		save_path = self.saver.save(self.sess, "../model/model" + self.name + ".ckpt")
		print("Model saved in file: %s" % save_path)

	def output_training_results(self):
		#print "Output Training Results"
		self.sampling = 1
		
		for self.step in range(1):
			self.cut_batch(0)
			
			pred = self.sess.run(self.pred, feed_dict = {
				self.x_nn: self.x_batch, self.y_nn: self.y_batch, 
				self.mask_nn: self.mask, self.acc_count: self.count_words, 
				self.bs: self.batch_size, self.keep_prob: self.drop})

			pred = self.sess.run(tf.nn.softmax(pred, -1))
			pred = self.sess.run(tf.argmax(pred, 2))
			print pred.shape
			print pred[0][0]
			
			for i in range(15):
				print str(i) + " " + str(self.step)
				self.fout_results.write(str(i) + " " + str(self.step) + " " + str(self.sampling) + "\n")
				if self.step == 0 or True:
					for j in range(self.num_steps-1):
						print self.id_word[self.y[i][j+1]], # check
						self.fout_results.write(self.id_word[self.y[i][j+1]] + " ")
					print ""
					self.fout_results.write("\n")
				for j in range(self.num_steps-1):
					print self.id_word[pred[i][j]],
					self.fout_results.write(self.id_word[pred[i][j]] + " ")
				print ""
				self.fout_results.write("\n")
			self.fout_results.write("\n\n")
		#fout.close()
	
	def load_test(self):
		self.x_test = seq2seq_lib.preprocess_testing_data(self.num_steps)
		print self.x_test
		if self.x_test.shape[0] <= 64:

			self.x_test_vocab = seq2seq_lib.one_hot(\
				self.x_test, self.x_test.shape[0], self.num_steps, self.vocab_size)
		else:
			self.x_test_vocab = seq2seq_lib.one_hot(\
				self.x_test[:64], 64, self.num_steps, self.vocab_size)

			self.x_test_vocab_1 = seq2seq_lib.one_hot(\
				self.x_test[64:], self.x_test.shape[0]-64, self.num_steps, self.vocab_size)

	def cut_batch_test(self):
		self.big = False
		if self.x_test.shape[0] <= 64:
			self.x_batch[:self.x_test.shape[0]] = self.x_test_vocab
		else:
			self.big = True
			self.x_batch_1 = self.x_batch[:]
			self.x_batch = self.x_test_vocab
			self.x_batch_1[:self.x_test.shape[0]-64] = self.x_test_vocab_1
			print self.x_batch_1.shape

		
	def testing(self):
		self.cut_batch_test()
		if not self.big:
			pred = self.sess.run(self.pred_test, feed_dict = {
				self.x_nn: self.x_batch, self.bs: 64, self.keep_prob: self.drop})

			#self.logProb(pred)
			pred = self.sess.run(tf.nn.softmax(pred, -1))	
			pred = self.sess.run(tf.argmax(pred, 2))
				
			
			
			
		else:
			pred = self.sess.run(self.pred_test, feed_dict = {
				self.x_nn: self.x_batch, self.bs: 64, self.keep_prob: self.drop})
			pred_1 = self.sess.run(self.pred_test, feed_dict = {
				self.x_nn: self.x_batch_1, self.bs: 64, self.keep_prob: self.drop})

			#self.logProb(pred)
			pred = self.sess.run(tf.nn.softmax(pred, -1))	
			pred = self.sess.run(tf.argmax(pred, 2))

			pred_1 = self.sess.run(tf.nn.softmax(pred_1, -1))	
			pred_1 = self.sess.run(tf.argmax(pred_1, 2))
				
			pred = np.concatenate((pred, pred_1))
			
		self.output(pred)
		self.output_real_test(pred)

		return pred
			
	def output(self, pred):
		for i in range(self.x_test.shape[0]):
			for j in range(self.num_steps):
				self.fout_test.write(self.id_word[self.x_test[i][j]] + " ")
			self.fout_test.write("\n")
			#for i in range(self.x_test.shape[0]+5):
			for j in range(self.num_steps):
				self.fout_test.write(self.id_word[pred[i][j]] + " ")
			self.fout_test.write("\n\n")
		self.fout_test.write("---------------------\n")

	def output_real_test(self, pred):
		
		for i in range(self.x_test.shape[0]):
			for j in range(self.num_steps):
				if pred[i][j] == 3:
					continue
				elif pred[i][j] == 2:
					break
				self.fout_tt.write(self.id_word[pred[i][j]] + " ")
			self.fout_tt.write("\n")


	def close_file(self):
		self.fout.close()
		self.fout_results.close()
		self.fout_test.close()
 
'''model = Model()
model.load()
model.set_variable()
model.set_container()
model.build_model()
model.load_test()
model.initialize_model()
model.summary()
model.training()
model.output_training_results()
model.testing()
model.close_file()'''

#model.ans_all()
print "\n Compile successfully!"

