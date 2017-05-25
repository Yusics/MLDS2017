import tensorflow as tf
import numpy as np
import model
#import argparse
import pickle
from os.path import join
#import h5py
#from Utils import image_processing
import scipy.misc
import random
import json
import os
import time


gpu_frac = 0.4
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
print "GPU fraction Usage: %.2f" % gpu_frac


def main():
	
	model_options = {
		'z_dim' : 100,
		't_dim' : 256,
		'batch_size' : 64,
		'image_size' : 64,
		'gf_dim' : 64,
		'df_dim' : 64,
		'gfc_dim' : 1024,
		'caption_vector_length' : 4800
	}
	beta1 = 0.5
	lr    = 2e-4
	z_dim = 100
	t_dim = 256
	batch_size = 64
	image_size = 64
	gfc_dim = 1024
	caption_vector_length = 4800
	epochs = 600
	
	
	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()
	
	checkpoint_dir = "./DCModel"


	d_optim = tf.train.AdamOptimizer(lr*0.5, beta1 = beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
	g_optim = tf.train.AdamOptimizer(lr, beta1 = beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])
	
	#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		tf.global_variables_initializer().run()
	
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state("Data/Models/")
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Model restored.")


		loaded_data = load_training_data()
		start_time = time.time()
		
		for i in range(epochs):
			#batch_no = 0
			for batch_no in range(loaded_data['data_length'] // batch_size):
			#while batch_no* < loaded_data['data_length']:
				real_images, wrong_images, caption_vectors, z_noise = get_training_batch(batch_no, batch_size, z_dim, loaded_data)
				
				# DISCR UPDATE
				check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
				_, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise
					})
				

				# GEN UPDATE
				

				# GEN UPDATE TWICE, to make sure d_loss does not go to 0
				_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})

				_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})

				_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})

				_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})


				_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})

				_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})

				_, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})

				
				
				#batch_no += 1
				#if batch_no == 171:
				#	continue
				if batch_no == 100:
					print "Saving Images, Model"
					#print "d1", d1
					#print "d2", d2
					#print "d3", d3
					#print "D", d_loss
					print "Epoch: [%2d], [%4d/%4d]  d_loss: %.8f, g_loss: %.8f, time: %4.4f" %(i, batch_no, len(loaded_data['image_list'])/ batch_size, d_loss, g_loss, 
					time.time() - start_time )
					save_for_vis(real_images, gen)
					saver.save(sess, checkpoint_dir)


			if i%3 == 0:
				save_path = saver.save(sess, "Data/Models/model_after_epoch_{}.ckpt".format(i))

def load_training_data():
	training_image_list = np.load("../hw3/image_feat.npy")
	captions = np.load("../hw3/embed_sent.npy")
	return {
			'image_list' : training_image_list,
			'captions' : captions,
			'data_length' : len(training_image_list)
		}
	

def save_for_vis(real_images, generated_images):

	#shutil.rmtree( join("./", 'samples') )
	#os.makedirs( join("./", 'samples') )

	for i in range(0, real_images.shape[0]):
		real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		real_images_255 = (real_images[i,:,:,:])
		scipy.misc.imsave( join("./sample_image", 'samples/{}.jpg'.format(i)) , real_images_255)

		fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		fake_images_255 = (generated_images[i,:,:,:])
		scipy.misc.imsave(join("./sample_image", 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)


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