import random
import numpy as np
import os

class Game(object):
	"""docstring for 2048"""
	def __init__(self, size):
		super(Game, self).__init__()
		self.size = size

		self.array = np.zeros([self.size, self.size])

		self.score = 0

		self.addRandomFigure(number=2)

		self.gridGame = self.array

	def is_finished(self):
		return 2048 in self.array

	def game_over(self):

		# self.cas = []

		for i in range(self.size):
			for j in range(self.size):
				if self.array[i][j] == 0:
					return False

				if j != self.size-1:
					if self.array[i][j] == self.array[i][j+1]:
						return False
				if i != self.size-1:
					if self.array[i][j] == self.array[i+1][j]:
						return False
				if j == self.size-1 and i != self.size-1:
					if self.array[i][j] == self.array[i+1][j]:
						return False
				if i == self.size-1 and j != self.size-1:
					if self.array[i][j] == self.array[i][j+1]:
						return False
		# self.cas5
		return True

	def reset(self):
		self.array = np.zeros([self.size, self.size])
		self.score = 0
		self.addRandomFigure(number=2)

	def generateFigure(self):
		if random.uniform(0,1) < 0.75:
			return 1
		return 2

	def freePosition(self):
		tabNull = []

		for i in range(self.size):
			for j in range(self.size):
				if self.array[i][j] == 0:
					tabNull.append([i, j])

		return tabNull

	def generatePosition(self):

		tabNull = self.freePosition()
		return random.choice(tabNull)

	def addRandomFigure(self, number=1):
		for i in range(number):
			position = self.generatePosition()
			self.array[position[0]][position[1]] = self.generateFigure()

	def moveUp(self):

		cpt, moved, merge = 0, False, True

		for j in range(self.size):
			for i in range(self.size):

				if(self.array[i][j] != 0):
					if cpt != 0 and self.array[i][j] == self.array[cpt-1][j] and merge:
						self.array[cpt-1][j] = self.array[i][j] + 1
						self.score += self.array[i][j] * 2
						self.array[i][j] = 0
						merge = False
						moved = True
					else:
						self.array[cpt][j] = self.array[i][j]
						if i != cpt:
							self.array[i][j] = 0
							moved = True
						cpt += 1
						merge = True
			merge = True
			cpt = 0

		if moved:
			self.addRandomFigure()
		# else:
		# 	self.score -= 5

	def moveDown(self):

		cpt, moved, merge = 0, False, True

		for j in range(self.size):
			for i in range(self.size-1, -1, -1):

				if(self.array[i][j] != 0):
					if cpt != 0 and self.array[i][j] == self.array[self.size-cpt][j] and merge:
						self.array[self.size-cpt][j] = self.array[i][j] + 1
						self.score += self.array[i][j] * 2
						self.array[i][j] = 0
						merge = False
						moved = True
					else:
						self.array[self.size-1-cpt][j] = self.array[i][j]
						if i != self.size-1-cpt:
							self.array[i][j] = 0
							moved = True
						cpt += 1
						merge = True
			merge = True
			cpt = 0

		if moved:
			self.addRandomFigure()
		# else:
		# 	self.score -= 5

	def moveLeft(self):

		cpt, moved, merge = 0, False, True

		for i in range(self.size):
			for j in range(self.size):

				if(self.array[i][j] != 0):
					if cpt != 0 and self.array[i][j] == self.array[i][cpt-1] and merge:
						self.array[i][cpt-1] = self.array[i][j] + 1
						self.score += self.array[i][j] * 2
						self.array[i][j] = 0
						merge = False
						moved = True
					else:
						self.array[i][cpt] = self.array[i][j]
						if j != cpt:
							self.array[i][j] = 0
							moved = True
						cpt += 1
						merge = True
			merge = True
			cpt = 0

		if moved:
			self.addRandomFigure()
		# else:
		# 	self.score -= 5

	def moveRight(self):

		cpt, moved, merge = 0, False, True

		for i in range(self.size):
			for j in range(self.size-1, -1, -1):

				if(self.array[i][j] != 0):
					if cpt != 0 and self.array[i][j] == self.array[i][self.size-cpt] and merge:
						self.array[i][self.size-cpt] = self.array[i][j] + 1
						self.score += self.array[i][j] * 2
						self.array[i][j] = 0
						merge = False
						moved = True
					else:
						self.array[i][self.size-1-cpt] = self.array[i][j]
						if j != self.size-1-cpt:
							self.array[i][j] = 0
							moved = True
						cpt += 1
						merge = True
			merge = True
			cpt = 0

		if moved:
			self.addRandomFigure()
		# else:
		# 	self.score -= 5

	def step(self, action):
		self.score = 0
		if action == 0:
			self.moveUp()
		elif action == 1:
			self.moveDown()
		elif action == 2:
			self.moveLeft()
		elif action == 3:
			self.moveRight()

		return self.score

		if self.is_finished():
			return self.score
		if self.game_over():
			return -1




if __name__ == '__main__':
	jeu = Game(4)
	print(jeu.array)
	while not jeu.game_over():
		direction = int(input("Direction: "))
		print(jeu.step(direction-1))
		print(jeu.array)
		

