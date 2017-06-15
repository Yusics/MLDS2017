#import sys
import pickle
#import string # punctuation
import numpy as np
#from string import digits # remove digit
#from numpy import genfromtxt
from collections import Counter # remove infrequent words
#coding: utf-8

'''Undo:
output to model/file
add other preprocess and check
'''

class Model(object):
	def __init__(self):
		self.conv_path = "../cornell movie-dialogs corpus/movie_conversations.txt"
		self.line_path = "../cornell movie-dialogs corpus/movie_lines.txt"

	def merge(self):
		self.conv_id = [] # store 2 pairs of line_id
		f = open(self.conv_path, "r")
		
		for i, line in enumerate(f):
			start = line.index("[")
			l = line[start+1:-2]
			l = [w for w in l.split("'")]
			n = len(l)/2 - 1 # number of pairs for this conversation

			for j in range(n):
				self.conv_id.append([str(l[2*j+1]), str(l[2*j+3])])
			
			if i % 20000 == 0: print i
			#print len(line)
			#if i > 1000: break

		#pickle.dump(self.line_dict, open("conv_id", "wb")) 
		f.close()

	def build_line_dict(self):
		'''f = open(self.line_path, "r")
		self.line_dict = {}
		for i, line in enumerate(f):
			l = line.split("+++$+++")
			#print l[0][:-1], l[4].strip()
			#print l[4]
			self.line_dict[l[0][:-1]] = l[4].strip()
			#if i > 10: break

		#print self.line_dict['L924']
		#print self.conv_id[0][0]
		#for i in range(5):
		#	print self.line_dict[self.conv_id[i][0]], self.line_dict[self.conv_id[i][1]]
		#exit()
		pickle.dump(self.line_dict, open("line_dict", "wb"))
		f.close()'''

		print "Loading Dictionary..."
		self.line_dict = pickle.load(open("line_dict", "rb"))
		print "Finished Loading."
 
	def add_line(self):
		self.conv.append(["How are you?", "I am fine."])
		self.conv.append(["See you later!", "Goodbye!"])
		self.conv.append(["Hello!", "Hi, nice to see you!"])
		self.conv.append(["What is your name?", "My name is Tiffany."])
		print len(self.conv[0])

	def change_to_line(self):
		print "Change To Line"
		self.conv = []
		self.add_line()
		i0 = len(self.conv)

		for i in range(len(self.conv_id)):
			self.conv.append([])
			#print self.line_dict[self.conv_id[i][0]], self.line_dict[self.conv_id[i][1]]
			self.conv[i+i0].append(self.line_dict[self.conv_id[i][0]])
			self.conv[i+i0].append(self.line_dict[self.conv_id[i][1]])
			#if i > 100: break
			if i % 20000 == 0: print i
		#print len(self.conv_id)
		
		'''for i in range(len(self.conv)):
			for j in range(2):
				print self.conv[i][j]
			print ""'''
		
	def preprocess(self): # can adjust to be faster
		print "\nStart Preprocessing\n"
		self.corpus = []
		#remove_list = []
		fout = open('../out.txt','w')
	
		other = ["<u>", "<\u>", "</u>", " v ", " q ", " .f ", " z ", " f ", " u "]
		# ([.,!?\"':;)(]) -> ()[]\
		print "conversation.len =", len(self.conv)
		for i in range(len(self.conv)):
			temp = []
			for j in range(2):
				#print self.conv[i][j]
				try: l = self.conv[i][j]
				except IndexError: print i, j, self.conv[i]
				#print l
				#print len(l)

				if len(l.split()) >= 1:
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

					#l = l.replace("---", " ")
					#l = l.replace("--", " ")
					#l = l.translate(string.maketrans("",""), string.punctuation) # remove punctuation
					#l = l.lower().translate(None, digits)
					l = l.lower()
					l = l.split() # 20?
					if i > 221610:
						print l
					for j in range(len(l)):
						try: 
							a = int(l[j])
							#if i > 221610: print w,
							l[j] = "Number"
							if i > 221610: print l
						except ValueError: continue
					#l = [s for s in l if s.isdigit()]
					
					#for w in l:
					#	print w, w[0], w[0].isupper()
						#if w[0].isupper() and w != "I" and w != "Mr" and w != "Mrs": 
						#	w = "unknownname"
					if i > 221610:
						print l

				else: break
				 
				'''fout.write(self.conv[i][j] + "\n-> ")
				for w in l:
					fout.write(w + " ")
				fout.write("\n")'''
				temp.append(l)
			#fout.write("\n")

			if len(temp) == 2: self.corpus.append(temp)

			if i % 20000 == 0: print i
			#if i > 100: break
		fout.close()
		print "corpus.len =", len(self.corpus)
		self.output_corpus()
		#for i in range(10):	print self.corpus[i]

	def output_corpus(self):
		fout = open('../data/corpus_for_check','w')
		for i in range(len(self.corpus)):
			for j in range(2):
				for k in range(len(self.corpus[i][j])):
					fout.write(str(self.corpus[i][j][k]) + " ")
				fout.write("\n")
			fout.write("\n")
			if i % 20000 == 0: print i
		fout.close()	

	def output_id_word(self):
		fout = open('../data/id_word_for_check','w')
		self.id_word[0] = "_"
		for i in range(len(self.id_word)):
			fout.write(str(i) + " " + self.id_word[i] + "\n")
		pickle.dump(self.id_word, open("../data/id_word_dict", "wb"))	
		#exit()
		fout.close()

	def count(self):
		word_set = []
		for i in range(len(self.corpus)): 
			for j in range(2):
				word_set += self.corpus[i][j]
		# change [[1, 2, 3], [4, 5]] into [1, 2, 3, 4, 5]

		self.common = [] # save the most common word in order
		self.word_id = {} 
		self.id_word = {}
		counts = Counter(word_set)
		commonTemp = counts.most_common(9996)
		for i in range(len(commonTemp)):
			#print commonTemp[i][1], commonTemp[i][0]
			self.common.append(commonTemp[i][0])
		for i, w in enumerate(self.common):
			self.word_id[w] = i+4
			self.id_word[i+4] = w
		self.word_id["BOS"] = 1
		self.word_id["EOS"] = 2
		self.word_id["Unknown"] = 3
		#self.word_id["Number"] = 4
		self.id_word[1] = "BOS"
		self.id_word[2] = "EOS"
		self.id_word[3] = "Unknown"
		#self.id_word[4] = "Number"
		self.output_id_word()


		print "Common.len = ", len(self.common)		
		
		self.common = np.array(self.common)
		print self.common[:10]
		#self.remove_uncommon()
		pickle.dump(self.common, open("../data/common_word_list", "wb"))
		pickle.dump(self.word_id, open("../data/word_id_dict", "wb"))
		#exit()
	def remove_uncommon(self):
		#fin = open("../result/add_to_common.txt", "r")
		count = 0
		for i in range(len(self.corpus)):
			for j in range(2):
				#print self.corpus[i][j]
				for w in self.corpus[i][j]:
					if w not in self.common:
						print w
						w = "Unknown"
						count += 1
		print count
		self.vocab_size = len(self.common)
		print "vocab = ", self.vocab_size()

	def change_to_id(self, max_length):

		corpus = np.zeros([len(self.corpus), 2, max_length], dtype = int)
		remove_list = []
		for i in range(len(self.corpus)):
			# for input message
			if len(self.corpus[i][0]) > max_length - 2: 
				#print len(self.corpus[i][0]) - 28, 
				m = int(len(self.corpus[i][0]) - max_length - 2)
				#print self.corpus[i][0][m:]
				self.corpus[i][0] = self.corpus[i][0][m:]
				#print len(self.corpus[i][0])
			
			#if i> 1000: break
			n = min(max_length - 2, len(self.corpus[i][0]))
		
			pad = max_length - (n+2)
			corpus[i][0][pad] = 1
			for k in range(n):
				try:
					corpus[i][0][pad+k+1] = self.word_id[self.corpus[i][0][k]]
				except KeyError:
					corpus[i][0][pad+k+1] = 3
			corpus[i][0][pad+n+1] = 2

			# for response
			n = min(max_length - 2, len(self.corpus[i][1]))
			corpus[i][1][0] = 1
			for k in range(n):
				try:
					corpus[i][1][k+1] = self.word_id[self.corpus[i][1][k]]
				except KeyError:
					corpus[i][1][k+1] = 3
			corpus[i][1][n+1] = 2

			# check
			for j in range(max_length-1):
				if corpus[i][1][j] == 3 and corpus[i][1][j+1] == 3:
					remove_list.append(i)

			if i % 20000 == 0: print i
		
		print "Removing unclean sentences."
		for i in range(len(remove_list)-1, -1, -1):
			corpus = np.delete(corpus, remove_list[i], axis = 0)
			if i % 100 == 0: print i

		
		#self.output_corpus(corpus)

		np.save("../data/corpus", corpus)

		for i in range(3):
			print self.corpus[300+i]
			print corpus[300+i], "\n"


model = Model()
model.merge()
model.build_line_dict()
model.change_to_line()
model.preprocess()
model.count()
model.change_to_id(15)
#model.save()

