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



	def random(self, n_features, n_classes, hidden_layer):
		n_hidden_layer = hidden_layer.length
		self.weights = []

		for i in range(0, n_hidden_layer+2):


		

	def fit(self, X, y):
		n_classes = y.nunique()
		random(X.shape[1], n_classes, self.hidden_layer)
