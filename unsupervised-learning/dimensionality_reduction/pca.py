import numpy as np

class PCA:
	def __init__(self, num_components=2):
		self.components = None
		self.eingenvalues = None
		self.eingenvectors = None
		self.num_components = num_components
	
	def __get_components(self):
		# sort components by eigenvalues
		pass
		
	def fit(self, X):
		# step 1 - compute mean row
		mean = np.mean(X, axis=0)
		ones = np.ones((X.shape[0], X.shape[1]))
		mean_bar = ones * mean
		# step 2 - substract mean
		B = X - mean_bar
		# step 3 - covariance of B
		C = np.cov(B.T)
		I = np.identity(X.shape[0])
		# step 4 - compute eigenvectors and eigenvalues of C
		self.eingenvalues, self.eingenvectors = np.linalg.eig(C)
		self.components = self.eingenvectors.T.dot(B.T)
			
if __name__ == '__main__':
	dataset = np.random.rand(1000, 2)
	model = PCA()
	model.fit(X=dataset)
