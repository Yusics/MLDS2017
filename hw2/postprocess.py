import numpy as np
import pickle
import json

f = open('src/sentence_test_all', 'r')


testing_id = pickle.load(open('testing_id', 'rb'))



ans = []
for i, line in enumerate(f):
	#if i % 2 == 1 :
	ans.append(line)
		#print i, line,
	#if i >= 100: break



def one_sentence(s):
	s = ans[i]
	print s,
	s = s.replace("BOS", "")
	s = s.replace("EOS", "")
	s = [w for w in s.split()]
	s[0] = s[0].capitalize()
	
	s = " ".join(s)

	return s

output_json = []

fout = open('ans.txt','w')
for i in range(len(ans)):
	test = dict()
	idx = testing_id[i]
	s = one_sentence(ans[i])
	test["caption"] = s
	test["id"] = idx
	output_json.append(test)
	fout.write(s + "\n")
	print s


with open("output.json", "w") as j:
	json.dump(output_json, j)




	