from collections import Counter
import math

def euclidian_distance_calc(point_x_list,point_y_list):
    sum_of_squared_distance=0
    for i in range(len(point_x_list)):
        sum_of_squared_distance+= math.pow((point_x_list[i]-point_y_list[i]),2)
    return math.sqrt(sum_of_squared_distance)

