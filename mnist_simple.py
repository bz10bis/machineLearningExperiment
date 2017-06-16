import time
start_time = time.time()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from Dbmanager import Dbmanager

import tensorflow as tf
import os
import argparse 

parser = argparse.ArgumentParser(description="Tweak hyper parameter for convnet")
parser.add_argument('-l', dest='learning_rate', action='store', type=float, default=0.0001, help='define learing rate')
parser.add_argument('-i', dest='iterations', action='store', type=int, default=1000, help='define number of iteration')
parser.add_argument('-b', dest='batch_size', action='store', type=int, default=50,  help='define the batch size')
args = parser.parse_args()
sess = tf.InteractiveSession()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dbm = Dbmanager()

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# define our model
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10]) #correct awnsers

#Represente les erreurs de notres modeles 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#on utilise la backpropagation avec un learning rate de 0.5 et on essayer de minimiser nos erreurs(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cross_entropy)

tf.global_variables_initializer().run()

for _ in range(args.iterations):
	batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
eval_result = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
end_time = time.time() - start_time
dbm.new_row(__file__, eval_result, end_time)