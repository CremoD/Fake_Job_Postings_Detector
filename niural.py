import numpy as np

class Niural:

	def __init__(self, n_epochs = 10, learning_rate = 0.001, max_iter= 100, hidden_layer = [100]):
		self.n_epochs = n_epochs
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.hidden_layer = hidden_layer

	def prova(self):
		print("Number of epochs", self.n_epochs)
		print("learning_rate", self.learning_rate)
		print("max_iter", self.max_iter)
		print("Fitting the data")
		

	def fit(self, X, y):
		n_classes = y.nunique()[0]
		
		if(n_classes == 2):
			n_classes = 1

		self.random_weights(X.shape[1], n_classes, self.hidden_layer)

		print("Weights")
		for mat in self.weights:
			print(mat)
			print("-------------------------")

		res = self.forward_propagate(X.loc[0,:])
			
		print("Prediction is:")
		print(res)

		self.train_errors = self.layers_errors(res,np.array(y.loc[0,:]))

			
		print("-------------------------")
		print(self.train_errors)

	def random_weights(self, n_features, n_classes, hidden_layer):
		self.weights = []
		
		units = [n_features] + hidden_layer + [n_classes]

		for i in range(0, len(units) - 1):
			self.weights.append(np.random.rand(units[i+1],units[i] + 1))

	def forward_propagate(self, x):
		a = x.to_numpy().reshape(-1,1)
		a_history = [a]

		for theta in self.weights:
		    a =  np.vstack((np.array([[1]]), a))
		    z = np.dot(theta, a)
		    a = self.sigmoid(z)
		    a_history.append(a)

		return a_history

	def layers_errors(self,a,y):
		# calculate error for output layer
		errors = [(a[-1] - y).tolist()]

		i = len(self.hidden_layer)
		while i >= 1:
			print("Looping over",i)
			curr_a = np.vstack((np.array([[1]]), a[i]))
			current_error = np.multiply(np.dot(np.transpose(self.weights[i]),np.array(errors[0])),np.multiply(curr_a, (1 - curr_a)))
			error = np.delete(current_error, 0, 0)
			errors.insert(0,error.tolist())
			i = i - 1

		return errors

	def sigmoid(self,number):
		return 1/(1 + np.exp(-number))

