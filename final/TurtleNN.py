import numpy as np
from tqdm import tqdm
import pandas as pd

class TNN:
	# constructor to take desired input parameters
	def __init__(self, n_epochs = 10, learning_rate = 0.1, reg_lambda = 0.001, hidden_layers = [100]):
		self.n_epochs = n_epochs
		self.learning_rate = learning_rate
		self.reg_lambda = reg_lambda
		self.hidden_layers = hidden_layers
		
		#print our logo
		turtle = (r'''
				     ___-------___
                                 _-~~             ~~-_
                             _-~                    /~-_
          /^\__/^\          /~  \                   /    \
         /|  O|| O|       /     \_______________/          \
        | |___||__|      /       /                \          \
        |          \    /      /                    \          \
        |   (_______) /______/                        \_________ \
        |         / /         \                      /             \
         \         \^\\         \                  /                 \     /
          \         ||           \______________/      _-_          //\__//
           \       ||------_-~~-_ ------------- \ --/~   ~\        || __/)
            ~-----||====/~      |==================|       |/~~~~~
             (_(__/  ./       /                   \_\      \.
                       (_(___/                       \_____)_)''')

		print(turtle)


		# printing used parameters
		print("\nTurtle Neural Network trained with the following parameters:")
		print("\tNumber of epochs:", self.n_epochs)
		print("\tLearning_rate:", self.learning_rate)
		print("\tRegularization factor:", self.reg_lambda)
		print("\tHidden layers configuration:", self.hidden_layers)
		

	#########################################################################
	# 							FIT METHOD									#
	#		Method used to train the Neural Network and obtain the best 	#
	#		possible weights. 												#								
	#########################################################################
	def fit(self, X, y):

		#check if input data are pandas dataframe as required
		if not isinstance(X, pd.DataFrame):
			raise Exception('Features data should be a Pandas DataFrame, now it is {0}'.format(type(X).__name__))

		if not isinstance(y, pd.DataFrame):
			raise Exception('Target data should be a Pandas DataFrame, now it is {0}'.format(type(y).__name__))

		n_classes = y.shape[1]

		#check if we have multiple classes but one hot encoding has not been applied
		if n_classes == 1 and y.nunique()[0] > 2:
			raise Exception('Make sure to apply one-hot encoding in case of multiple classes')

		#initialize weights randomly
		self.random_weights(X.shape[1], n_classes, self.hidden_layers)

		#iterate over all epochs
		for j in range(0, self.n_epochs):
			self.epochs_errors = 0
			#initialize the progress bar with the desired format
			epochString = "Epoch 0{0}".format(j) if j < 10 else "Epoch {0}".format(j) 
			pbar = tqdm(total=X.shape[0], desc=epochString, bar_format="{l_bar}{bar}{postfix}")
			#for each training instance
			for i in X.index:
				#forward propagation 
				self.a_history = self.forward_propagate(X.loc[i,:])
				
				#we get training error for this instance
				self.train_errors = self.back_propagate(self.a_history,np.array(y.loc[i,:]))
				
				#we add it to the aggregated MSE for this epoch
				epoch_err = np.array(self.train_errors[-1]).flatten()
				epoch_err = np.absolute(epoch_err)
				self.epochs_errors = self.epochs_errors + np.square(np.mean(epoch_err))
				
				# update weights
				for i in range(0, len(self.weights)):
					update_mat = self.update_weights(self.weights[i], self.a_history[i], self.train_errors[i]) 
					self.weights[i] = self.weights[i] - (self.learning_rate*update_mat)
				pbar.update(1)

			#print MSE after the progress bar
			mse = self.epochs_errors/X.shape[0]
			pbar.postfix = "MSE: {:.6f}".format(mse)
			pbar.close()



	#########################################################################
	# 								PREDICT 								#
	# 		Method used to predict the class of the given instances			#
	#					  with the trained parameters						#
	#########################################################################
	def predict(self, X):

		y_pred = []
		#for each of the given instances
		for i in X.index:
			#forward propagate
			actual_a = self.forward_propagate(X.loc[i,:])
			#if we have multiple output neurons
			if actual_a[-1].shape[0] > 1:
				#we take the index of the maximum probability
				index = np.argmax(actual_a[-1])
				#and we set it to one with all the others 0s
				res = np.zeros(actual_a[-1].flatten().shape)
				res[index] = 1
				y_pred.append(res)
			#if there is only one output neuron
			else:
				#take the probability and compare with threshold
				prob = actual_a[-1][0][0]
				
				if prob >= 0.5:
					y_pred.append(1)
				else:
					y_pred.append(0)

		return y_pred


	#########################################################################
	# 							UPDATE WEIGHTS 								#
	# 				We update the weights based on the errors 				#
	#					of the current training instance 					#
	#########################################################################
	def update_weights(self, theta, a, errors):
		 #we add a 1 for the bias
		 a =  np.vstack((np.array([[1]]), a))
		 #initialize empty gradients
		 gradients = np.array([])

		 #for each error
		 for err in errors:
		 	#compute current gradient
		 	curr_grad = err * a
		 	gradients = np.concatenate((gradients, curr_grad.flatten()))

		 gradients_mat = gradients.reshape(theta.shape)

		 #apply regularisation to non-bias terms
		 gradients_mat[:,1:] = gradients_mat[:,1:] + theta[:,1:] * self.reg_lambda	 

		 return gradients_mat






	#########################################################################
	# 						 	RANDOM WEIGHTS 								#
	# 				Initialize matrix of weights randomly 					#
	#########################################################################
	def random_weights(self, n_features, n_classes, hidden_layers):
		self.weights = []
		#we compute the number of units taking in consideration
		#the neurons of the input and output layer
		units = [n_features] + hidden_layers + [n_classes]

		for i in range(0, len(units) - 1):
			self.weights.append(np.random.rand(units[i+1],units[i] + 1))


	#########################################################################
	# 						FORWARD PROPAGATION 							#
	# 			Method to forward propagate a single instance 				#
	#########################################################################
	def forward_propagate(self, x):
		#convert to numpy and reshape into a column
		a = x.to_numpy(dtype = np.float32).reshape(-1,1)
		a_history = [a]
		#for each layer weights
		for theta in self.weights:
			#add bias term
		    a =  np.vstack((np.array([[1]]), a))
		    #apply standard forward propagation
		    z = np.dot(theta, a)
		    a = self.sigmoid(z)
		    a_history.append(a)

		return a_history

	#########################################################################
	# 							BACK PROPAGATION							#
	# 		Compute the errors of each layer using backpropagation 			#
	#########################################################################
	def back_propagate(self,a,y):
		# calculate error for output layer
		errors = [(a[-1] - y.reshape(-1,1)).tolist()]
		i = len(self.hidden_layers)
		# for each layer but the first one
		while i >= 1:
			#add bias term
			curr_a = np.vstack((np.array([[1]]), a[i]))
			#compute current error using backpropagation
			current_error = np.multiply(np.dot(np.transpose(self.weights[i]),np.array(errors[0])),np.multiply(curr_a, (1 - curr_a)))
			#delete bias error
			error = np.delete(current_error, 0, 0)
			errors.insert(0,error.tolist())
			i = i - 1

		return errors

	#########################################################################
	# 							SIGMOID FUNCTION 							#
	# 	Function used to compute the sigmoid value of the given number 		#
	#########################################################################
	def sigmoid(self,number):
		return 1/(1 + np.exp(-number))

