import numpy as np
import json
import sys
import os
import re
import pickle


test_id = sys.argv[1]
test_feature = sys.argv[2]

#TEXT_DATA_DIR = "MLDS_hw2_data/training_data/feat/"


#all_label = []
#id_to_index = []

#with open("MLDS_hw2_data/training_label.json") as json_data:
#	label = json.load(json_data)

#with open("MLDS_hw2_data/testing_public_label.json") as json_test:
#	test_label = json.load(json_test)



#test_all_label =[] 
test_id_to_index = []

with open(test_id, 'r') as f:
	for line in f.readlines():
		line = line.strip()
		test_id_to_index.append(line)

pickle.dump(test_id_to_index, open("testing_id", "wb"))

	
#for i in range(len(label)):
#	all_label.append(label[i]["caption"])
#	id_to_index.append(label[i]["id"])

'''all_feat = np.empty([len(label), 80, 4096], dtype=np.float32)

for feat in os.listdir(TEXT_DATA_DIR):
	idx = feat.strip().split(".npy")[0]
	path = os.path.join(TEXT_DATA_DIR, feat)
	try:
		idx = id_to_index.index(idx)
		data = np.load(path)
		all_feat[idx] = data
		
	except:
		continue



for i in range(len(test_label)):
	test_all_label.append(test_label[i]["caption"])
	test_id_to_index.append(test_label[i]["id"])


pickle.dump(test_id_to_index, open("test_id_to_index", "wb"))'''


test_all_feat = np.empty([len(test_id_to_index), 80, 4096], dtype=np.float32)

for feat in os.listdir(test_feature):
	idx = feat.strip().split(".npy")[0]
	path = os.path.join(test_feature, feat)
	try:
		idx = test_id_to_index.index(idx)
		data = np.load(path)
		test_all_feat[idx] = data
	except:
		continue

#print np.load("MLDS_hw2_data/testing_data/feat/ScdUht-pM6s_53_63.avi.npy")[0]

#print test_all_feat[0][0]


print test_all_feat.shape



np.save("feat_test.npy",test_all_feat)
#print test_all_feat
exit()







word_to_index = dict()
word_list = []

all_label_preprocess = []

max_length = 0

test_all_label_preprocess = []


for caption in all_label:
	process_sent = []
	for sent in caption:
		sent = re.sub('[^a-zA-Z\']'," ", sent)

		#test = sent.lower()
		#print test
		sent = sent.lower().split()
		if len(sent) > max_length:
			max_length = len(sent)
		for word in sent:
			if word not in word_list:
				word_to_index[word] = len(word_list)
				word_list.append(word)



		sent.insert(0,"BOS")
		sent.append("EOS")
		process_sent.append(sent)

	all_label_preprocess.append(process_sent)

#print len(word_list)
#exit()

#print max_length
#exit()

word_to_index["BOS"] = len(word_list) + 0
word_to_index["EOS"] = len(word_list) + 1
word_list.append("BOS")
word_list.append("EOS")

for caption in test_all_label:
	process_sent = []
	for sent in caption:
		sent = re.sub('[^a-zA-Z\']'," ", sent)

		#test = sent.lower()
		#print test
		sent = sent.lower().split()
		if len(sent) > max_length:
			max_length = len(sent)
		#for word in sent:
		#	if word not in word_list:
		#		word_to_index[word] = len(word_list)
		#		word_list.append(word)



		sent.insert(0,"BOS")
		sent.append("EOS")
		process_sent.append(sent)

	test_all_label_preprocess.append(process_sent)







all_label_preprocess_idx = []

for caption in all_label_preprocess:
	process_sent_idx = []
	#for sent in caption:
	sent_idx = []
	[sent_idx.append(word_to_index[word]) for word in caption[0]]
		#process_sent_idx.append(sent_idx)

	#all_label_preprocess_idx.append(process_sent_idx)
	all_label_preprocess_idx.append(sent_idx)




pickle.dump(word_list, open("word_list", "wb"))
pickle.dump(word_to_index, open("word_to_index", "wb"))
pickle.dump(all_label_preprocess_idx, open("all_label_preprocess_idx", "wb"))

#pickle.dump(id_to_index, open("id_to_index", "wb"))
#pickle.dump(all_label_preprocess, open("caption", "wb"))		






