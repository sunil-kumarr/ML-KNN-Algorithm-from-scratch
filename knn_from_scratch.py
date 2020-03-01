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

def main():
    '''
    # Regression Data
    #
    # Column 0: height (inches)
    # Column 1: weight (pounds)
    '''
    reg_data = [
       [65.75, 112.99],
       [71.52, 136.49],
       [69.40, 153.03],
       [68.22, 142.34],
       [67.79, 144.30],
       [68.70, 123.30],
       [69.80, 141.49],
       [70.01, 136.46],
       [67.90, 112.37],
       [66.49, 127.45],
    ]

    # Question:
    # Given the data we have, what's the best-guess at someone's weight if they are 60 inches tall?
    reg_query = [60]
    reg_k_nearest_neighbors, reg_prediction = knn_classifier(
        reg_data, reg_query, k_neighbors=3, distance_func=euclidian_distance_calc, choice_func=mean
    )

    '''
    # Classification Data
    #
    # Column 0: age
    # Column 1: likes pineapple
    '''
    clf_data = [
       [22, 1],
       [23, 1],
       [21, 1],
       [18, 1],
       [19, 1],
       [25, 0],
       [27, 0],
       [29, 0],
       [31, 0],
       [45, 0],
    ]
    # Question:
    # Given the data we have, does a 33 year old like pineapples on their pizza?
    clf_query = [33]
    clf_k_nearest_neighbors, clf_prediction = knn_classifier(
        clf_data, clf_query, k_neighbors=3, distance_func=euclidian_distance_calc, choice_func=mode
    )

if __name__ == '__main__':
    main()