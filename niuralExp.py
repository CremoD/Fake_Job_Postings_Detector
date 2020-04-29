import numpy as np

class Niural:

	def __init__(self, n_epochs = 10, learning_rate = 0.1, reg_lambda = 1, hidden_layer = [100]):
		self.n_epochs = n_epochs
		self.learning_rate = learning_rate
		self.reg_lambda = reg_lambda
		self.hidden_layer = hidden_layer

	def prova(self):
		print("Number of epochs", self.n_epochs)
		print("learning_rate", self.learning_rate)
		print("reg_lambda", self.reg_lambda)
		print("Fitting the data")
		

	#########################################################################
	# Fit method
	#########################################################################
	def fit(self, X, y):
		n_classes = y.nunique()[0]
		
		if(n_classes == 2):
			n_classes = 1

		self.random_weights(X.shape[1], n_classes, self.hidden_layer)

		for j in range(0, self.n_epochs):
			self.epochs_errors = 0
			for i in X.index:
				#print("Index", i)

				self.a_history = self.forward_propagate(X.loc[i,:])
					
				self.train_errors = self.layers_errors(self.a_history,np.array(y.loc[i,:]))
				self.epochs_errors = self.epochs_errors + np.square(self.train_errors[-1][0][0])
				# update weights
				for i in range(0, len(self.weights)):
					update_mat = self.update_weights(self.weights[i], self.a_history[i], self.train_errors[i]) 
					self.weights[i] = self.weights[i] - (self.learning_rate*update_mat)
			mse = self.epochs_errors/X.shape[0]
			print("Epoch: ", j, " , MSE: ", mse)

		print("----------------------")
		#for weig in self.weights:
		#	print(weig)



	#########################################################################
	# Predict
	#########################################################################
	def predict(self, X):

		y_pred = []

		for i in X.index:
			actual_a = self.forward_propagate(X.loc[i,:])
			#print(i, ":", actual_a, "\n------------------\n")
			prob = actual_a[-1][0][0]
			#print(prob)
			if prob >= 0.5:
				y_pred.append(1)
			else:
				y_pred.append(0)

		return y_pred


	#########################################################################
	# Update weights
	#########################################################################
	def update_weights(self, theta, a, errors):
		 a =  np.vstack((np.array([[1]]), a))
		 gradients = np.array([])

		 for err in errors:
		 	curr_grad = err * a
		 	gradients = np.concatenate((gradients, curr_grad.flatten()))

		 gradients_mat = gradients.reshape(theta.shape)

		 gradients_mat[:,1:] = gradients_mat[:,1:] + theta[:,1:] * self.reg_lambda	 

		 return gradients_mat






	#########################################################################
	# Random weight initialization
	#########################################################################
	def random_weights(self, n_features, n_classes, hidden_layer):
		self.weights = []
		
		units = [n_features] + hidden_layer + [n_classes]

		for i in range(0, len(units) - 1):
			self.weights.append(np.random.rand(units[i+1],units[i] + 1))


	#########################################################################
	# Forward propagation
	#########################################################################
	def forward_propagate(self, x):
		a = x.to_numpy(dtype = np.float32).reshape(-1,1)
		a_history = [a]

		for theta in self.weights:
		    a =  np.vstack((np.array([[1]]), a))
		    z = np.dot(theta, a)
		    a = self.sigmoid(z)
		    a_history.append(a)

		return a_history

	#########################################################################
	# layers error calculation
	#########################################################################
	def layers_errors(self,a,y):
		# calculate error for output layer
		errors = [(a[-1] - y).tolist()]

		i = len(self.hidden_layer)
		while i >= 1:
			curr_a = np.vstack((np.array([[1]]), a[i]))
			current_error = np.multiply(np.dot(np.transpose(self.weights[i]),np.array(errors[0])),np.multiply(curr_a, (1 - curr_a)))
			error = np.delete(current_error, 0, 0)
			errors.insert(0,error.tolist())
			i = i - 1

		return errors

	#########################################################################
	# Sigmoid function
	#########################################################################
	def sigmoid(self,number):
		return 1/(1 + np.exp(-number))

