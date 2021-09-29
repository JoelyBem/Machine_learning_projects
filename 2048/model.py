import tensorflow as tf
import numpy as np
from game_2048 import Game
from random import randint

def model_loss(model, Qtarget):
	loss = tf.reduce_mean(tf.square(model - Qtarget))
	return loss

def get_data(game, size):

	rewards = []
	states = []
	# data = open("data.txt", "a")
	# target = open("target.txt", "a")

	for _ in range(size):
		if game.game_over():
			game.reset()

		index = randint(0, len(states))

		# Stockage temporaire de jeu pour les 4 mouvements 
		array = np.copy(game.array)
		reward = []

		# states.insert(index, game.array.reshape(4,4,1))
		states.insert(index, game.array)

		for i in range(1):
			reward.append(game.step(i))
			game.array = np.copy(array)

		# data.write(str(game.array))
		# data.write("\n")
		# target.write(str(reward))
		# target.write('\n')
		
		game.step(randint(0,3))

		rewards.insert(index,reward)



	return states, rewards

def create_model():
	inputs = tf.placeholder(tf.float32, shape=[None, game.size, game.size])

	# conv = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[2,2], activation=tf.nn.relu)
	# conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[2,1], activation=tf.nn.relu)
	# conv = tf.layers.conv2d(inputs=conv, filters=128, kernel_size=[2,1], activation=tf.nn.relu)
	# print(conv)
	# return 1,1

	hidden = tf.layers.dense(inputs=tf.contrib.layers.flatten(inputs), units=8, activation=tf.nn.relu)

	hidden2 = tf.layers.dense(inputs=tf.contrib.layers.flatten(hidden), units=4, activation=tf.nn.relu)
	model = tf.layers.dense(inputs=hidden2, units=1)

	return inputs, model

if __name__ == '__main__':

	game = Game(4)

	inputs, model = create_model()
	sess = tf.Session()
	
	size = 2000

	states, rewards = get_data(game, size)	

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

	losses = []

	Qtarget = tf.convert_to_tensor(np.array(rewards), tf.float32)

	correct_prediction = tf.equal(tf.argmax(model), tf.argmax(Qtarget))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	batch = 32
	cpt = 0
	for i in range(0, size, batch):
		# break
		if i + batch < size:
			to = batch
			to_np = i + batch
		else:
			to = size - i
			to_np = size
		Qtarget_b = tf.slice(Qtarget, [i, 0], [to, Qtarget.shape[1]])
		states_b = np.array(states)[i:to_np]


		data = {
			inputs: states_b
		}

		sess.run(tf.global_variables_initializer())
		# print(sess.run(model, feed_dict=data))
		# break

		loss = tf.reduce_mean(tf.square(model - Qtarget_b))
		train = optimizer.minimize(loss)
		
		
		sess.run(train, feed_dict=data)
		losses = []
		losses.append(sess.run(loss, feed_dict=data))
		# print(sess.run(tf.reduce_mean(model), feed_dict=data), sess.run(tf.reduce_mean(Qtarget_b)))
		cpt += 1 
		if cpt%10:
			print("loss:",np.mean(np.array(losses)))
		print("accuracy:", sess.run(accuracy, feed_dict=data))

