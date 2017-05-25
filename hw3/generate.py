import tensorflow as tf
import numpy as np
import model
import pickle
from os.path import join
import scipy.misc
import random
import os
import time
import sys
import skipthoughts


gpu_frac = 0.3
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
print "GPU fraction Usage: %.2f" % gpu_frac


def main():
	
	
	beta1 = 0.5
	lr    = 2e-4
	z_dim = 100
	t_dim = 256
	batch_size = 64
	image_size = 64
	gfc_dim = 1024
	caption_vector_length = 4800
	epochs = 600
	path = sys.argv[1]

	test_data, test_id = load_test_data(path)
	embed_model = skipthoughts.load_model()
	caption_vectors = skipthoughts.encode(embed_model, test_data)

	#np.save("test_embedd.npy", caption_vectors)
	#exit()
	#caption_vectors = np.load("test_embedd.npy")
	caption_vectors = np.tile(caption_vectors, (5,1))


	model_options = {
		'z_dim' : 100,
		't_dim' : 256,
		'batch_size' : len(test_data)*5,
		'image_size' : 64,
		'gf_dim' : 64,
		'df_dim' : 64,
		'gfc_dim' : 1024,
		'caption_vector_length' : 4800
	}

	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()
	
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		tf.global_variables_initializer().run()
	
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state("Data/Models/")
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Model restored.")


		
		z_noise = np.random.uniform(-1, 1, [5*len(test_data), z_dim])

		gen = sess.run(outputs['generator'],
				feed_dict = {		
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise})			
		print "Saving Images, Model"
		
					
		save_image(gen, test_id)
					


		


def load_test_data(path):
	test_data = []
	test_id   = []

	with open(path) as f:
		for line in f.readlines():
			line = line.strip().split(",")
			print line[1]
			test_data.append(line[1])
			test_id.append(line[0])

	return test_data, test_id





def save_image(generated_images, test_id):
	size = len(test_id)
	
	for ro in range(5):
		for i in range(size):
			scipy.misc.imsave( "./samples/sample_{}_{}.jpg".format(test_id[i],ro), generated_images[ro*size+i])
	




def get_training_batch(batch_no, batch_size, z_dim, loaded_data = None):
	
	batch_idxs = loaded_data['data_length'] // batch_size

	real_images = loaded_data['image_list'][batch_no*batch_size:(batch_no+1)*batch_size]
	while True:
		wrong = random.randint(0, batch_idxs-1)
		if wrong != batch_no:
			break
	#if batch_no == batch_idxs-1:
	wrong_images = loaded_data['image_list'][wrong*batch_size:(wrong+1)*batch_size]
	#else:
	#	wrong_images = loaded_data['image_list'][(batch_no+1)*batch_size:(batch_no+2)*batch_size]
	captions = loaded_data['captions'][batch_no*batch_size:(batch_no+1)*batch_size]

	z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

	return real_images, wrong_images, captions, z_noise




if __name__ == '__main__':
	main()