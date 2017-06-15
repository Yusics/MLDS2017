import tensorflow as tf
import numpy as np
import pickle
import sys
import random
import seq2seq_lib
import math
from test_RL_512 import Model
 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config = config))


if sys.argv[1] == "S2S":
	model_path = "./S2S/model_seq2seq512"
else:
	model_path = "./RL/model-512-RL"

model = Model(model_path)
model.load()
model.set_variable()
model.set_container()
model.build_model()
model.load_test()
model.initialize_model()
model.summary()
#model.training()
#model.output_training_results()
model.testing()
model.close_file()




