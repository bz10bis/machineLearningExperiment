import time
start_time = time.time()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from Dbmanager import Dbmanager

import tensorflow as tf
import os 
import argparse

parser = argparse.ArgumentParser(description="Tweak hyper parameter for convnet")
parser.add_argument('-i', dest='iterations', action='store', type=int,  default=1000, help='define number of iterations')
parser.add_argument('-b', dest='batch_size', action='store', type=int, default=50,  help='define the batch size')
parser.add_argument('-l', dest='starter_learning_rate', action='store', type=float, default=0.01, help='define the starter learning rate for momentum')
parser.add_argument('-n', dest='nesterov', action='store', type=int, default=0, help='use nesterov')
parser.add_argument('-m', dest='momentum', action='store', type=float, default=0.96, help='set momentum value')
args = parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.InteractiveSession()
dbm = Dbmanager()

use_nesterov = False
if(args.nesterov==1):
	use_nesterov = True


x = tf.placeholder(tf.float32, [None, 784]) #inputs
y_ = tf.placeholder(tf.float32, [None, 10]) #desired outputs

W = tf.Variable(tf.zeros([784, 10])) #Weights: tensor full of zeros
b = tf.Variable(tf.zeros([10])) #bias same

sess.run(tf.global_variables_initializer()) #initialize variable in current session using default values

y = tf.matmul(x, W) + b # regression

#define our loss function (here cross entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#training the model 
train_step  = tf.train.MomentumOptimizer(args.starter_learning_rate, args.momentum, False, 'Momentum', use_nesterov ).minimize(cross_entropy)
sess.run(tf.global_variables_initializer()) 
for _ in range(args.iterations):
	batch = mnist.train.next_batch(args.batch_size) #take batch of 100 exemples 
	train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #feed placeholder with data

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
eval_result = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
end_time = time.time() - start_time
dbm.new_row(__file__, eval_result, end_time)



