import numpy as np

corpus = np.load("../data/" + "corpus_15.npy")
print "corpus =", corpus.shape
num_steps = 15
count = 0

for i in range(corpus.shape[0]):
	try:
		n = np.where(corpus[i, 1] == 3)[0][0]
		#print n
		#print corpus[i][1]
		for j in range(n, num_steps-1):
			corpus[i][1][j] = corpus[i][1][j+1]
		corpus[i][1][j+1] = 0
		#print corpus[i][1]
		count += 1
	except IndexError: 
		continue
	#if i > 10: break
print count
np.save("../data/corpus_15_remove", corpus)
