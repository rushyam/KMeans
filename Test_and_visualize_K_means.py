import numpy as np
import k_means
from k_means import k_means
x = np.random.rand(1000, 2)*100

kmeans = k_means(number_of_clusters = 5, number_of_iteration = 10, init = 'random')
points_with_centroid_index, centroids = kmeans.fit(x)
kmeans.visualize()

kmeans = k_means(number_of_clusters = 5, number_of_iteration = 10, init = 'initial_centroids_from_points')
points_with_centroid_index, centroids = kmeans.fit(x)
kmeans.visualize()