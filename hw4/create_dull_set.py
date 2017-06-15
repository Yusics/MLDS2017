import pickle
import numpy as np

dull_temp = [
	"I don't know.",
	"I have to go.",
	"no, I dont",
	"no, I dont know what you are talking about.",
	"I dont want to see you."
	]

word_id = pickle.load(open("../data/word_id_dict", "rb"))
dull = []
other = ["<u>", "<\u>", "</u>", " v ", " q ", " .f ", " z ", " f ", " u "]

for l in dull_temp:
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
	dull.append(l)
		
print "dull set.len =", len(dull)

#print word_id["i"]
print dull
# change word to id
dull_arr = np.zeros([len(dull), 15], dtype = int)
		
for i in range(len(dull)):
	n = min(13, len(dull[i]))
	dull_arr[i][0] = 1
	for k in range(n):
		try:
			dull_arr[i][k+1] = word_id[dull[i][k]]
		except KeyError:
			dull_arr[i][k+1] = 3
	dull_arr[i][n+1] = 2

'''for j in range(15):
	dull_arr[len(dull)+0][j] = 0
	dull_arr[len(dull)+1][j] = 0
	dull_arr[len(dull)+2][j] = 3
for i in range(3):
	dull_arr[len(dull)+i][0] = 1
	dull_arr[len(dull)+i][14] = 2'''


print dull_arr
np.save("../data/dull", dull_arr)
print "Compile Succesfully!"	


