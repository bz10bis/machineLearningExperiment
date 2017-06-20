import os 
import os.path 
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weights_variables(shape):
	#tuncated_normal return a random value
	initial_var = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial_var)

def bias_variables(shape):
	#Define bias constant
	initial_var = tf.constant(1, shape=shape)
	return tf.Variable(initial_var)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	#get the output of conv2d and downsample it
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

def init_summaries():
	tf.summary.histogramm("weights", W_conv1)
	tf.summary.histogramm("biases", b_conv1)
	tf.summary.histogramm("activations", h_conv1)
	tf.summary.image("inputs_images", reshaped_image)
	summary_op = tf.summary.merge_all()

def model():
	with tf.scope_name('Inputs'):
		x = tf.placeholder(tf.float32, [None, 784])
		y_ = tf.placeholder(tf.float32, [None, 10])

	with tf.scope_name('Conv1'):
		with tf.scope_name('Weights'):
			W_conv1 = weight_variable([5, 5, 1, 32])
 		with tf.scope_name('Bias'):
 			b_conv1 = bias_variable([32])
 		with tf.scope_name('Image'):
 			reshaped_image = tf.reshape(x, [-1, 28, 28, 1])
 		with tf.scope_name('Convolution'):
 			h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
			h_pool1 = max_pool_2x2(h_conv1)
 			

if __name__ == '__main__':

	#define root folder for tensorboard logs
	logs_path = 'Logs/'

	#get command line arguments and parse them
	parser = argparse.ArgumentParser(description="Launch tensorflow convolutional network")
	parser.add_argument('-l', dest='learning_rate', action='store', type=float, default=0.0001, help='define learning rate')
	parser.add_argument('-k', dest='keep_prob', action='store', type=float, default=0.5, help='define keep prob for dropout')
	parser.add_argument('-i', dest='iterations', action='store', type=int, default=1000, help='define number of iteration')
	parser.add_argument('-b', dest='batch_size', action='store', type=int, default=50,  help='define the batch size')
	args = parser.parse_args()

	#init timer
	start_time = time.time()

