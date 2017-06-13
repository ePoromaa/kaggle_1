from __future__ import print_function
import tensorflow as tf 
import tensorflow.contrib.slim as slim
import numpy as np 
from csv import DictReader
# from matplotlib import pyplot as plt
import random
from scipy import misc
from time import time
from sklearn.metrics import fbeta_score, recall_score, precision_score


def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
	return fbeta_score(y_true, y_pred, beta=2, average='samples'), recall_score(y_true, y_pred), precision_score(y_true,
																												 y_pred)


def optimise_f2_thresholds(y, p, verbose=False, resolution=100):
	""" Brute force over threshold for output variables """
	def mf(x):
		p2 = np.zeros_like(p)
		for i in range(17):
			p2[:, i] = (p[:, i] > x[i]).astype(np.int)
		score = fbeta_score(y, p2, beta=2, average='samples')
		return score

	x = [0.2]*17
	for i in range(17):
		best_i2 = 0
		best_score = 0
		for i2 in range(resolution):
			i2 /= float(resolution)
			x[i] = i2
			score = mf(x)
			if score > best_score:
				best_i2 = i2
				best_score = score
		x[i] = best_i2
		if verbose:
			print(i, best_i2, best_score)

	return x


def read_data():
	unique_labels = set()
	for row in DictReader(open('train.csv')):
		labels = row['tags'].split(' ')
		for label in labels:
			unique_labels.add(label)

	unique_labels = list(unique_labels)
	unique_labels = sorted(unique_labels)
	unique_labels = dict(zip(unique_labels,range(len(unique_labels))))
	print(unique_labels)
	return unique_labels


def extract_data(rows, batch_size, unique_labels):
	""" Generator for extracting data """
	image_batch = []
	label_batch = []
	#  global unique_labels
	for row in rows:
		filename = 'train-jpg/'+row['image_name']+'.jpg'
		image = misc.imread(filename,mode='RGB')
		image = image[:,:,0:3]
		image = image / 255.0

		labels = row['tags'].split(' ')
		label_vector = np.zeros(len(unique_labels))

		for label in labels:
			label_vector[unique_labels[label]] = 1.0

		image_batch.append(image)
		label_batch.append(label_vector)

		if len(image_batch) == batch_size:
			yield np.array(image_batch), np.array(label_batch)
			image_batch = []
			label_batch = []


def bn(x):
	""" Batch normalization, """
	return slim.batch_norm(x, scale=False, center=False, is_training=True)


def cnn(x):
	# The graph
	# Initial downsampling # 77 conv + maxpool
	hidden = slim.conv2d(x, 32, [7, 7], stride=2, padding="SAME", activation_fn=tf.nn.relu)
	hidden = slim.max_pool2d(hidden, [2, 2], stride=2)

	# 3x3 layers
	hidden = slim.conv2d(hidden,32,[3,3],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)

	# Block 1, begin by stride 2
	hidden = slim.conv2d(hidden,64,[3,3],stride=2,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,64,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,64,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,64,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)

	# Block 2, begin by stride 3
	hidden = slim.conv2d(hidden,128,[3,3],stride=2,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,128,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,128,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,128,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)

	# Block 3, begin by stride 3
	hidden = slim.conv2d(hidden,256,[3,3],stride=2,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,256,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,256,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)
	hidden = slim.conv2d(hidden,256,[1,1],stride=1,padding="SAME",activation_fn=tf.nn.relu,normalizer_fn=bn)

	# Global avg pool
	hidden = slim.avg_pool2d(hidden,kernel_size=[8,8],stride=1,padding="VALID")

	# FC layers
	hidden = slim.flatten(hidden)
	output_layer = slim.fully_connected(hidden,17,activation_fn=None)

	output_sigmoids = tf.sigmoid(output_layer)
	return output_sigmoids, output_layer


def main():
	x_placeholder = tf.placeholder(tf.float32, [None, 256, 256, 3])
	y_placeholder = tf.placeholder(tf.float32, [None, 17])
	num_params = 0
	l2_loss = 0.0
	output_sigmoids, output_layer = cnn(x_placeholder) # Create cnn

	for variable in tf.trainable_variables():
		print(variable)
		l2_loss += tf.reduce_sum(tf.square(variable))
		shape = variable.get_shape()
		n = 1
		for dim in shape:
			n *= dim.value
		num_params += n
	print("Number of parameters", num_params)

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_placeholder, logits=output_layer))
	learning_rate = tf.Variable(0.001, trainable=False)
	optimizer = tf.train.AdamOptimizer(learning_rate)

	train_op = optimizer.minimize(loss + 0.00005 * l2_loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	rows = [row for row in DictReader(open('train.csv'))]
	random.shuffle(rows)
	train_partition = int(len(rows) * 0.7)
	train_rows = rows[0:train_partition]
	validation_rows = rows[train_partition:len(rows)]

	print('The number of training rows', len(train_rows))
	UNIQUE_LABELS = read_data()
	best_epoch = 0
	best_loss = 0.0
	for epoch in range(100):

		train_loss = 0.0
		train_count = 0.0
		random.shuffle(train_rows)
		if epoch > 0 and epoch % 50 == 0:
			pass  # sess.run(tf.assign(learning_rate, learning_rate * 0.1))
		t1 = time()
		for i, (image_batch, label_batch) in enumerate(extract_data(train_rows, 256, UNIQUE_LABELS)):
			_, L = sess.run([train_op, loss], feed_dict={x_placeholder: image_batch, y_placeholder: label_batch})
			train_loss += L
			train_count += 1.
		elapsed_time = time() - t1

		validation_losses = []
		predictions = []
		targets = []
		for image_batch, label_batch in extract_data(validation_rows, 256, UNIQUE_LABELS):
			s, validation_loss = sess.run([output_sigmoids, loss],
										  feed_dict={x_placeholder: image_batch, y_placeholder: label_batch})

			validation_losses.append(validation_loss)
			targets.append(label_batch)
			predictions.append(s)

		predictions = np.concatenate(predictions)
		targets = np.concatenate(targets)

		thresholds = optimise_f2_thresholds(targets, predictions)
		p = np.zeros_like(predictions)
		for i in range(17):
			p[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int)
		fbeta, recall, precision = fbeta_score(targets, p, beta=2, average='samples')
		if fbeta > best_loss:
			best_loss = fbeta
			best_epoch = epoch
		with open("results.txt", "a") as outputfile:
			outputfile.write("Epoch "+ str(epoch)+
				  " Train="+str(train_loss / train_count)+
			  	" Test="+ str(sum(validation_losses) / len(validation_losses))+
			  	" F2="+ str(fbeta)+ " at "+ ",".join(map(str, thresholds))+
				" recall="+str(recall)+
				" precision"+str(precision)+
			  	" Time="+ str(elapsed_time)+
			  	" L2="+ str(sess.run(l2_loss))+"\n")
		if best_epoch + 5 < epoch:
			break

	return None


if __name__ == '__main__':
	print("Executed {0}.".format(__file__))
	main()
	print("END")


