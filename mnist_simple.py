import argparse 
parser = argparse.ArgumentParser(description="Tweak hyper parameter for convnet")
parser.add_argument('-l', dest='learning_rate', action='store', type=float, default=0.0001, help='define learing rate')
parser.add_argument('-i', dest='iterations', action='store', type=int, default=1000, help='define number of iteration')
parser.add_argument('-b', dest='batch_size', action='store', type=int, default=50,  help='define the batch size')
parser.add_argument('-r', dest='run_number', action='store', type=int, default=0,  help='define run number')
args = parser.parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

log_path = 'Logs'

import time
start_time = time.time()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)

from Dbmanager import Dbmanager

import tensorflow as tf

sess = tf.InteractiveSession()

dbm = Dbmanager()

with tf.name_scope('Inputs'):
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10]) #correct

with tf.name_scope('Weights'):	
	W = tf.Variable(tf.zeros([784, 10]))

with tf.name_scope('Biases'):
	b = tf.Variable(tf.zeros([10]))

with tf.name_scope('Model'):
	# define our model
	y = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope('Cross_entropy'):
	#Represente les erreurs de notres modeles 
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

with tf.name_scope('Train_step'):
	#on utilise la backpropagation avec un learning rate de 0.5 et on essayer de minimiser nos erreurs(cross_entropy)
	train_step = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path + '/simple' + '/' + str(args.run_number), sess.graph)

tf.global_variables_initializer().run()

for i in range(args.iterations):
	batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
	_, summary = sess.run([train_step, summary_op], feed_dict={x: batch_xs, y_:batch_ys})
	writer.add_summary(summary, i)

eval_result = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
end_time = time.time() - start_time
dbm.new_row(__file__, eval_result, end_time)