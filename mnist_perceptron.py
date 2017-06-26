import time
start_time = time.time()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)

from Dbmanager import Dbmanager

import tensorflow as tf
import os 
import io
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random as ran

parser = argparse.ArgumentParser(description="Tweak hyper parameter for perceptron")
parser.add_argument('-i', dest='iterations', action='store', type=int,  default=1000, help='define number of iterations')
parser.add_argument('-b', dest='batch_size', action='store', type=int, default=100,  help='define the batch size')
parser.add_argument('-l', dest='learning_rate', action='store', type=float, default=0.01, help='define the starter learning rate for momentum')
parser.add_argument('-r', dest='run_number', action='store', type=int, default=0,  help='define run number')

args = parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#logs_path = 'Logs/perceptron/'
logs_path = 'Logs/perceptron/' + str(args.run_number) + '/'
sess = tf.InteractiveSession()
dbm = Dbmanager()

def get_heatmap():
	for i in range(10):
		plt.subplot(2, 5, i+1)
		weight = sess.run(W)[:,i]
		buf = io.BytesIO()
		plt.figure
		plt.title(i)
		plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
		frame1 = plt.gca()
		frame1.axes.get_xaxis().set_visible(False)
		frame1.axes.get_yaxis().set_visible(False)
	plt.savefig(buf, format='png')
	buf.seek(0)
	return buf

# use_nesterov = False
# if(args.nesterov==1):
# 	use_nesterov = True

with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, 784], name='inputs') #inputs
	y_ = tf.placeholder(tf.float32, [None, 10], name='desired_outputs') #desired outputs

with tf.name_scope('weight'):
	W = tf.Variable(tf.zeros([784, 10]), name='Weights') #Weights: tensor full of zeros

with tf.name_scope('biases'):
	b = tf.Variable(tf.zeros([10]), name='Bias') #bias same

with tf.name_scope('Model'):
	y = tf.matmul(x, W) + b # regression

#define our loss function (here cross entropy)
with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('train'):
	#training the model
	train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(cross_entropy)
	#train_step = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cross_entropy)
	#train_step = tf.train.AdadeltaOptimizer(args.learning_rate,0.95,1e-08).minimize(cross_entropy)
	#train_step = tf.train.AdagradOptimizer(args.learning_rate).minimize(cross_entropy)
	#train_step = tf.train.MomentumOptimizer(args.learning_rate, 0.9, False, 'Momentum', True ).minimize(cross_entropy)

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
# image_reshaped = tf.reshape(x, [-1, 28, 28, 1])
# tf.summary.image('inputs', image_reshaped, 10)
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logs_path + '/' + str(args.run_number), sess.graph)

sess.run(tf.global_variables_initializer())
#writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())  
#writer = tf.summary.FileWriter(logs_path, sess.graph)  
for i in range(args.iterations):
	batch = mnist.train.next_batch(args.batch_size) #take batch of 100 exemples 
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	_, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1]}, options=run_options, run_metadata=run_metadata)
	if i % 100 == 0:		
		plot_buf = get_heatmap()
		image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
		image = tf.expand_dims(image, 0)
		predict_summary = tf.summary.image("plot", image)
		new_summary = sess.run(predict_summary,options=run_options, run_metadata=run_metadata)
		train_writer.add_run_metadata(run_metadata, 'step%d' % i)
		train_writer.add_summary(new_summary, i)
	#train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #feed placeholder with data
	train_writer.add_summary(summary, i)

print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.validation.images, y_: mnist.validation.labels}, session=sess))
# print(sess.run(tf.argmax(y,1), feed_dict={x: mnist.test.images}))
# print(tf.argmax(mnist.test.labels,1))
end_time = time.time() - start_time
print("End in :" + str(end_time))
#dbm.new_row(__file__, eval_result, end_time)
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     weight = sess.run(W)[:,i]
#     plt.title(i)
#     plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
#     frame1 = plt.gca()
#     frame1.axes.get_xaxis().set_visible(False)
#     frame1.axes.get_yaxis().set_visible(False)
# plt.show()



def display_compare(num):
    x_train = mnist.test.images[num,:].reshape(1,784)
    y_train = mnist.test.labels[num,:]
    label = y_train.argmax()
    prediction = sess.run(y, feed_dict={x: x_train}).argmax()
    buf = io.BytesIO()
    plt.figure()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# plot_buf = get_heatmap()
# image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# image = tf.expand_dims(image, 0)
# predict_summary = tf.summary.image("plot", image)
# new_summary = sess.run(predict_summary)
# writer_img = tf.summary.FileWriter('Logs/momentum/images', sess.graph)
# writer_img.add_summary(new_summary)
# writer_img.close()

# def display_compare(num):
#     x_train = mnist.test.images[num,:].reshape(1,784)
#     y_train = mnist.test.labels[num,:]
#     label = y_train.argmax()
#     prediction = sess.run(y, feed_dict={x: x_train}).argmax()
#     plt.title('Prediction: %d Label: %d' % (prediction, label))
#     plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
#     plt.show()

# # display_compare(ran.randint(0, 5000))
# display_compare(2)
# res = correct_prediction.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# for i in range(5000):
# 	if(res[i] == False):
# 		display_compare(i)