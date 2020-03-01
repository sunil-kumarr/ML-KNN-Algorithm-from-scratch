from collections import Counter
import math

def euclidian_distance_calc(training_data_points,query_data_point):
    sum_of_squared_distance=0
    for i in range(len(training_data_points)):
        sum_of_squared_distance+= math.pow((training_data_points[i]-query_data_point[i]),2)
    return math.sqrt(sum_of_squared_distance)


def mean(labels):
    sum(labels)/len(labels)


def mode(labels):
    Counter(labels).most_common(1)[0][0]


def knn_classifier(train_data_points, query_point, k_neighbors,distance_func,choice_func):
    neighbor_distances_and_indices=[]
    for index,training_feature in enumerate(train_data_points):
        #! pass columns (0 : n-1) i.e. feature datapoints except the target feature and test point
        distance = distance_func(training_feature[:-1],query_point)
        neighbor_distances_and_indices.append((distance,index))
    #! sort in ascending based on distance of points
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    #! pick first nearest "k" neighbors from the sorted list
    k_nearest_neighbors = sorted_neighbor_distances_and_indices[:k_neighbors]
    #! find the labels of the nearest points in the training data_set
    k_nearest_neighbors_labels = [train_data_points[index][1] for distance,index in k_nearest_neighbors]
    #! Based on choice_function used it can be both regression and classification model
    #!  mean function: KNN-Regressor
    #!  mode function : KNN-Classifier
    return k_nearest_neighbors,choice_func(k_nearest_neighbors_labels)
