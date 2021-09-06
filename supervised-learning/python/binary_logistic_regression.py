import numpy as np

class LogisticRegression:

	def __init__(self, alpha=0.1, epochs=100, convergence=0.0001):
		self.alpha = alpha
		self.epochs = epochs
		self.convergence = convergence
		self.W = None
		self.b = None
		self.J = 0

	def __activation_function(self, x):
		return 1.0 / (1.0 + np.exp(-x))

	def __loss_function(self, y, a):
		return -(y * np.log(a) + (1 - y) * np.log(1 - a))
	
	def fit(self, X, y):
		self.W = np.array(np.random.rand(X.shape[1]))
		self.b = 0.01
		epoch = 0
		old_J = 0
		new_J = 1
		while epoch < self.epochs and np.abs(old_J - new_J) > self.convergence:
			old_J = new_J
			Z = X.dot(self.W.T) + self.b # W.T * X + b
			A = [self.__activation_function(z) for z in Z]
			dZ = A - y
			new_J = np.sum([self.__loss_function(y=y, a=a) for y, a in zip(y, A)]) / X.shape[0]
			dw = dZ.dot(X) / X.shape[0] # dw = (X * dZ.T) / X.shape[0]
			db = np.sum(dZ) / X.shape[0] # db = dZ / Z.shape[0]
			self.W = self.W - self.alpha * dw
			self.b = self.b - self.alpha * db
			print('Epoch: {0} - cost function: {1}'.format(epoch, new_J))
			self.J = new_J
			epoch += 1		
		
	def predict(self, X):
		y_hat = [self.__activation_function(x.dot(self.W.T) + self.b) for x in X]
		return [1 if pred > 0.5 else 0 for pred in y_hat]
	
if __name__ == '__main__':
	model = LogisticRegression(epochs=10000)
	size = 100
	dataset = np.random.rand(size, 2)
	y = np.random.randint(2, size=size)
	model.fit(dataset, y)
	X_test = np.random.rand(10, 2)
	print(model.predict(X=X_test))
