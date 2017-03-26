from numpy import genfromtxt
import numpy as np
import pickle
import csv
import sys

class Model(object):
	def __init__(self):
		self.N_max = 30
		self.N_maxWord = 30
		self.back = 2
		self.file_name = sys.argv[1]

	def cut(self):
		fin = self.file_name
		with open(fin, "r") as f:
			raw_data = csv.reader(f)
			raw_data = list(raw_data)
			raw_data = raw_data[1:len(raw_data)]
		raw_data = np.array(raw_data)

		print "raw_data.shape = ", raw_data.shape
		#print raw_data[1]

		self.corpus = raw_data[:, 1]
		self.choice = raw_data[:, 2:]

		#pickle.dump(self.corpus, open("x_test_word", "wb"))
		#pickle.dump(self.choice, open("y_test_word", "wb"))
	
	def load_xy(self):
		self.corpus = pickle.load(open("x_test_word", "rb"))
		self.choice = pickle.load(open("y_test_word", "rb"))

	def load(self):
		#self.dict = pickle.load(open("dict", "rb"))
		self.common1 = pickle.load(open("common1_word", "rb"))

	def pre_sentence(self):
		print "Sentences Preprocessing"

		temp = []
		for i in range(len(self.corpus)):
			l = self.corpus[i]
			l = l.lower()
			l = l.replace("-", " ")
			l = l.replace(".", " ")
			l = l.replace(",", " ")
			l = l.replace(":", " ")
			l = l.replace(";", " ")
			l = l.replace("[", " ")
			l = l.replace("]", " ")
			l = [w for w in l.split()]
			temp.append(l)
		self.corpus = temp
		print "Corpus len = ", len(self.corpus)

	def remove_infrequent(self):
		print "Remove Infrequent Words"
		self.common1.append("_____")

		for i in range(len(self.corpus)):
			self.corpus[i] = [w for w in self.corpus[i] if w in self.common1]
		#self.remove_short(self.corpus)

		'''fout = open(self.path + "result/temp2", "wb")
		for i in range(200):
			fout.write("%s\n" %self.corpus[i]) '''

	def get_pos(self):
		print "Get Position"
		self.pos = np.empty((len(self.corpus)), dtype = int)
		for i in range(len(self.corpus)):
			try:
				temp = self.corpus[i].index("_____")
				self.pos[i] = temp
				#if i<10: print temp
	
			except ValueError: 
				continue
				#print "Error!", i
				
	'''def output(self, txt, name):
		fout = open(self.path + "/result/temp" + str(name), "wb")
		for i in range(len(txt)):
			fout.write("%s\n" % txt[i])'''

	def x_test(self):
		run_again = 1
		if run_again == 1:
			self.pre_sentence()
			self.remove_infrequent()
			self.get_pos()
			#pickle.dump(self.corpus, open("x_test_word2", "wb"))
			#pickle.dump(self.pos, open(self.path + "model/pos", "wb"))
		else:
			self.corpus = pickle.load(open("x_test_word2", "rb"))
			self.pos = pickle.load(open("pos", "rb"))

	def x_test_hot(self):
		print "x_test Processing"
		#self.corpus = pickle.load(open("x_test_word2", "rb"))
		self.dict = pickle.load(open("my_wordid", "rb"))
		self.x_test = np.zeros([1040, self.N_maxWord], dtype = int)
		for i in range(1040):
			front = self.pos[i]
			if front <= self.N_maxWord:
				pad = self.N_maxWord - front # start from here
				for j in range(front): 
					try:
						self.x_test[i][j+pad] = self.dict[self.corpus[i][j]] + 1
					except KeyError:
						continue
						#print self.corpus[i][j]
			else:
				pad = front - self.N_maxWord
				for j in range(self.N_maxWord): 
					try:
						self.x_test[i][j] = self.dict[self.corpus[i][j+pad]] + 1
					except KeyError:
						continue
						#print self.corpus[i][j]
			'''if i == 100:
				print front
				print pad
				print self.corpus[i]
				print self.x_test[i]'''
		pickle.dump(self.x_test, open("x_test_hot", "wb"))
		print "x_test = ", self.x_test.shape

	def y_test_hot(self):
		self.dict = pickle.load(open("choice_list", "rb"))
		#self.choice = pickle.load(open("y_test_word", "rb"))
		self.y_test = np.zeros([len(self.choice), 5])
		total = 0
		for i in range(len(self.choice)):
			for j in range(5):
				temp = self.choice[i][j]
				temp = temp.lower()
				temp = temp.replace("-", "")
				
				try:
					index = self.dict.index(temp)
					self.y_test[i][j] = index
				except ValueError:
					continue
					#print i, j, self.choice[i][j], temp
		
		print "y_test = ", self.y_test.shape
		#print self.y_test[100]
		pickle.dump(self.y_test, open("y_test_hot", "wb"))

	'''def build_y_test(self):
		print "y_test Processing"
		self.choice_dict = pickle.load(open("choice_list", "rb"))
		#self.choice = pickle.load(open("y_test_word", "rb"))
		self.y_test_id = np.empty([len(self.choice), 5], dtype = int)
		#print self.choice.shape

		for i in range(len(self.choice)):
			for j in range(5):
				temp = self.choice[i][j]
				temp = temp.lower()
				temp = temp.replace("-", "")
				try:
					index = self.choice_dict.index(temp)
					self.y_test_id[i][j] = index
					print index
				except ValueError:
					self.y_test_id[i][j] = 0
					#print i, j, self.choice[i][j]
		print "y_test = ", self.y_test_id.shape
		#print self.y_test_id[0:1]
		pickle.dump(self.y_test_id, open("y_test_id2", "wb"))'''


model = Model()

run_again = 1
if run_again == 1: 
	model.cut()
else: 
	model.load_xy()
model.load()

model.x_test()
model.x_test_hot()
model.y_test_hot()

