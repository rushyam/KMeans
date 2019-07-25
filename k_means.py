import numpy as np 
import math
import matplotlib.pyplot as plt

def euclidean_distance(point, centroid):
	return math.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)

def find_new_centroids(points_in_centroids, n_clusters):
	centroids = np.zeros((n_clusters, 2), dtype = float)
	index = 0

	for centroid, points in points_in_centroids.items():
		for point in points:
			centroids[index][0] += point[0]
			centroids[index][1] += point[1]
		centroids[index][0] = centroids[index][0]/len(points)
		centroids[index][1] = centroids[index][1]/len(points)
		index += 1

	return centroids

def initialize_centroid(n_clusters, x, init):
	if init == 'initial_centroids_from_points':
		indx = np.random.randint(0, len(x), n_clusters)
		return x[indx, : ]

	if init == 'random':
		bottom_left_point = np.amin(x, axis=0)
		top_right_point = np.amax(x, axis=0)
		
		centroids_1st_value = np.random.uniform(bottom_left_point[0], top_right_point[0], n_clusters)
		centroids_2nd_value = np.random.uniform(bottom_left_point[1], top_right_point[1], n_clusters)
		centroids = np.zeros((n_clusters, 2), dtype = float)
		for i in range(n_clusters):
			centroids[i][0] = centroids_1st_value[i]
			centroids[i][1] = centroids_2nd_value[i]
		return centroids

class k_means():
	def __init__(self, number_of_clusters, number_of_iteration = 10, init = 'initial_centroids_from_points'):
		if number_of_clusters <= 0:
			raise ValueError('Number of clusters should be greater than 0.')
		if number_of_iteration <= 0:
			raise ValueError('Number of iteration should be greater than 0.')
		if init != 'initial_centroids_from_points' and init != 'random':
			raise ValueError('init should be "initial_centroids_from_points" or "random".')

		self.n_clusters = number_of_clusters
		self.max_iter = number_of_iteration
		self.visualize_flage = False
		self.init = init

	def fit(self, X):
		if X.shape[1] != 2:
			raise ValueError('Dimension is not correct.')
		self.data = X
		centroids = initialize_centroid(self.n_clusters, X, self.init)
		
		number_of_iteration = 0
		infinity = float("inf")
		points_with_centroid_index = []

		while(number_of_iteration < self.max_iter):
			
			points_in_centroids = {}
			
			for point in X:
				distance_of_point_from_centroid = infinity
				centroid_of_point = None

				for centroid in centroids:
					distance = euclidean_distance(point, centroid)
					if distance_of_point_from_centroid > distance:
						distance_of_point_from_centroid = distance
						centroid_of_point = centroid

				if str(centroid_of_point) not in points_in_centroids.keys():
					points_in_centroids[str(centroid_of_point)] = []
				points_in_centroids[str(centroid_of_point)].append(point)

			centroids = find_new_centroids(points_in_centroids, self.n_clusters)

			number_of_iteration += 1
			
			if number_of_iteration >= self.max_iter:
				for point in X:
					distance_of_point_from_centroid = infinity
					index = -1;
					count = 0
					for centroid in centroids:
						distance = euclidean_distance(point, centroid)
						if distance_of_point_from_centroid > distance:
							distance_of_point_from_centroid = distance
							index = count
						count += 1
					points_with_centroid_index.append(index)

		self.visualize_flage = True
		self.cluster_id_of_points = points_with_centroid_index
		self.final_centroids = centroids
		return points_with_centroid_index, centroids

	def visualize(self):
		if self.visualize_flage == False:
			raise ValueError('visualize method called before fit method.')

		plt.scatter(self.data[:, 0], self.data[:, 1], c=self.cluster_id_of_points, s=50, cmap='viridis')
		plt.scatter(self.final_centroids[:, 0], self.final_centroids[:, 1], c='black', s=200, alpha=0.5)
		plt.show()

