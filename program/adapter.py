import pylab
import RDA
import numpy as np

def ret_nan_coors(np_arr):
    points = list()

    l = len(np_arr)
    for i in range(l):
        for j in np.where(np.isnan(np_arr[i]))[0]:
            points.append([i, j])

    return points


def ret_coors(np_arr):
    points = list()

    l = len(np_arr)
    for i in range(l):
        for j in np.where(~np.isnan(np_arr[i]))[0]:
            points.append([i, j, np_arr[i][j]])

    return points


def count_p(arr):
    points = ret_coors(arr)

    rda = RDA.RDA(2, 100, 0.9)

    for el in points:
        rda.add_point(el[:2], el[2])

    rda.do_approx()

    unknown_points = ret_nan_coors(arr)
    counter = 0
    for el in unknown_points:
        counter += 1
        counted = rda.count(RDA.Location(el))
        arr[el[0]][el[1]] = counted