import numpy as np


def re_ordering(image, size):
    w_state = image[0][size - 1][0][0] - image[0][0][0][0]
    h_state = image[0][-1][0][1] - image[0][0][0][1]

    # true right or up or horizontal  false left or down or vertical

    if w_state > 0:
        right = False
    else:
        right = True

    if h_state > 0:
        up = False
    else:
        up = True

    w_state = image[0][1][0][0] - image[0][0][0][0]
    h_state = image[0][1][0][1] - image[0][0][0][1]

    if w_state > h_state:
        horizontal = True
    else:
        horizontal = False

    if not right and not up and not horizontal:
        object_point = np.zeros((size * size, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:size, 0:size].T.reshape(-1, 2)

    elif not right and not up and horizontal:
        object_point = np.zeros((size * size, 3), np.float32)
        x, y = np.mgrid[0:size, 0:size]
        n = np.array([y, x])
        n = n.T.reshape(-1, 2)
        object_point[:, :2] = n
        object_point = [object_point]

    elif not right and up and not horizontal:
        object_point = np.zeros((size * size, 3), np.float32)
        n = np.mgrid[0:size, 0:size].T
        for s in range(0, len(n)):
            n[s] = np.sort(n[s], axis=0)[::-1]
        object_point[:, :2] = n.reshape(-1, 2)
        object_point = [object_point]

    elif not right and up and horizontal:
        object_point = np.zeros((size * size, 3), np.float32)
        x, y = np.mgrid[0:size, 0:size]
        n = np.array([y, x])
        n = n.T
        n = n[::-1]
        object_point[:, :2] = n.reshape(-1, 2)
        object_point = [object_point]

    elif right and up and horizontal:
        object_point = np.zeros((size * size, 3), np.float32)
        x, y = np.mgrid[0:size, 0:size]
        n = np.array([y, x])
        n = n.T
        n = n[::-1]
        for s in range(0, len(n)):
            n[s] = np.sort(n[s], axis=0)[::-1]
        object_point[:, :2] = n.reshape(-1, 2)
        object_point = [object_point]

    elif right and up and not horizontal:
        object_point = np.zeros((size * size, 3), np.float32)
        n = np.mgrid[0:size, 0:size]
        n = n.T
        n = n[::-1]
        for s in range(0, len(n)):
            n[s] = np.sort(n[s], axis=0)[::-1]
        object_point[:, :2] = n.reshape(-1, 2)
        object_point = [object_point]

    elif right and not up and not horizontal:
        object_point = np.zeros((size * size, 3), np.float32)
        n = np.mgrid[0:size, 0:size]
        n = n.T
        n = n[::-1]
        object_point[:, :2] = n.reshape(-1, 2)
        object_point = [object_point]

    else:
        object_point = np.zeros((size * size, 3), np.float32)
        x, y = np.mgrid[0:size, 0:size]
        n = np.array([y, x])
        n = n.T
        for s in range(0, len(n)):
            n[s] = np.sort(n[s], axis=0)[::-1]
        object_point[:, :2] = n.reshape(-1, 2)
        object_point = [object_point]

    return object_point
