import numpy as np
import math


def homogeneous_coordinate(vector):
    return np.array([vector[0], vector[1], 1])


def de_homogeneous_coordinate(vector):
    return np.array([vector[0], vector[1]])


def to_normal_plane(inverse, vector):
    return np.dot(inverse, vector)


def distort_normal(vector, distortion_values):
    try:
        dist = distortion_values[0]
        r2 = vector[0] ** 2 + vector[1] ** 2
        radial_d = 1 + dist[0] * r2 + dist[1] * r2 ** 2 + dist[4] * r2 ** 3
        x_d = radial_d * vector[0] + 2 * dist[2] * vector[0] * vector[1] + dist[3] * (r2 + 2 * vector[0] ** 2)
        y_d = radial_d * vector[1] + dist[2] * (r2 + 2 * vector[1] ** 2) + 2 * dist[3] * vector[0] * vector[1]
        return np.array([x_d, y_d])
    except:
        return np.array([0, 0])


def un_distort(vector, distortion_values, threshold):
    p_d = vector
    p_u = vector
    while True:
        dist = distort_normal(p_u, distortion_values)
        err = (dist - p_d)
        p_u = (p_u - err)
        if (math.fabs(err[0]) < threshold) and (math.fabs(err[1]) < threshold):
            break
    return p_u


def to_realworld_coordinate(inverse, vector, transfer_vector):
    zero = np.array([0, 0, 0])
    transfer_vector = np.array(transfer_vector, dtype='f')
    transfer_vector = transfer_vector.reshape(-1, 1).transpose()
    t_zero = zero - transfer_vector[0]
    t_vector = vector - transfer_vector[0]
    realworld_zero = np.dot(inverse, t_zero)
    realworld_vector = np.dot(inverse, t_vector)

    dr = realworld_vector - realworld_zero
    k = -(realworld_zero[2] / dr[2])
    re = [realworld_zero[0] + k * dr[0], realworld_zero[1] + k * dr[1], realworld_zero[2] + k * dr[2]]
    return re


def translate_to_realworld_coordinate(vector, inverse_matrix, inv_rot_vec_matrix,
                                      transfer_vector, distortion_value, threshold):
    vector = homogeneous_coordinate(vector)
    vector = to_normal_plane(inverse_matrix, vector)
    vector = de_homogeneous_coordinate(vector)
    vector = un_distort(vector, distortion_value, threshold)
    vector = homogeneous_coordinate(vector)
    vector = to_realworld_coordinate(inv_rot_vec_matrix, vector, transfer_vector)
    return vector
