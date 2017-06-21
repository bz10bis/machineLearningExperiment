import os
import time
import io
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import argparse
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")
if not (os.path.isfile(LABELS) and os.path.isfile(SPRITES)):
  print("Necessary data files were not found. Run this command from inside the "
    "repo provided at "
    "https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial.")
  exit(1)

def convolution_layer(layer_input, input_size, output_size, name="convolutional"):
	with tf.name_scope(name):
		W = tf.Variable(tf.truncated_normal([5,5, input_size, output_size], stddev=0.1), name="Weights")
		b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="Bias")
		conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding="SAME")
		activation_function = tf.nn.relu(conv + b)
		tf.summary.histogram("weights", W)
		tf.summary.histogram("biases", b)
		tf.summary.histogram("activations", activation_function)
		return tf.nn.max_pool(activation_function, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def dense_layer(layer_input, input_size, output_size, name="dense"):
	with tf.name_scope(name):
		W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name="Weights")
		b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="Bias")
		activation_function = tf.matmul(layer_input, W) + b
		tf.summary.histogram("weights", W)
		tf.summary.histogram("biases", b)
		tf.summary.histogram("activations", activation_function)
		return activation_function
	

def model(learning_rate, parameters):
	tf.reset_default_graph()
	sess = tf.Session()
	x = tf.placeholder(tf.float32, shape=[None, 784], name="X")
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	tf.summary.image('input', x_image, 3)
	y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
	conv1 = convolution_layer(x_image, 1, 32, "conv1")
	conv2 = convolution_layer(conv1, 32, 64, "conv2")
	#flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])
	flatten_output = tf.reshape(conv2, [-1, 7 * 7 * 64])
	dense_layer1 = dense_layer(flatten_output, 7 * 7 * 64, 1024, "dense_layer1")
	relu = tf.nn.relu(dense_layer1)
	tf.summary.histogram("dense_layer1/relu", relu)
	embedding_input = relu
	embedding_size = 1024
	logits = dense_layer(dense_layer1, 1024, 10, "logit_layer")
	with tf.name_scope("cross_entropy"):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="cross_entropy")
		tf.summary.scalar("cross_entropy", cross_entropy)
	with tf.name_scope("train"):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	with tf.name_scope("accuracy"):
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar("accuracy", accuracy)
	summary_op = tf.summary.merge_all()
	embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
	assignement = embedding.assign(embedding_input)
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter(logs_path + parameters)
	writer.add_graph(sess.graph)
	config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
	embedding_config = config.embeddings.add()
	embedding_config.tensor_name = embedding.name
	embedding_config.sprite.image_path = SPRITES
	embedding_config.metadata_path = LABELS
	embedding_config.sprite.single_image_dim.extend([28, 28])
	tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
	for i in range(2001):
		batch = mnist.train.next_batch(100)
		if i % 5 == 0:
			[train_accuracy, summary] = sess.run([accuracy, summary_op], feed_dict={x: batch[0], y: batch[1]})
			writer.add_summary(summary, i)
		if i % 500 == 0:
			sess.run(assignement, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
			saver.save(sess, os.path.join(logs_path, "model.ckpt"), i)
		sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def parameters_to_string(learning_rate):
	return "lr_%.0E" % (learning_rate)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Tweak hyper parameter for convnet")
	parser.add_argument('-i', dest='iterations', action='store', type=int,  default=1000, help='define number of iterations')
	parser.add_argument('-b', dest='batch_size', action='store', type=int, default=50,  help='define the batch size')
	parser.add_argument('-l', dest='starter_learning_rate', action='store', type=float, default=0.01, help='define the starter learning rate for momentum')
	parser.add_argument('-n', dest='nesterov', action='store', type=int, default=0, help='use nesterov')
	parser.add_argument('-m', dest='momentum', action='store', type=float, default=0.96, help='set momentum value')
	parser.add_argument('-r', dest='run_number', action='store', type=int, default=0,  help='define run number')
	args = parser.parse_args()

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	logs_path = 'Logs/momentum2/'

	start_time = time.time()

	for learning_rate in [1E-3, 1E-4]:
	#for learning_rate in [0.01]:
		parameters = parameters_to_string(learning_rate)
		model(learning_rate, parameters)
	end_time = time.time() - start_time 
	print("Learning Finished in: " + str(end_time))