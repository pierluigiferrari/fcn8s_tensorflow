import numpy as np

def convert_IDs_to_IDs(input_array, id_map_array):
    '''
    Converts an array of integers to an array of the same shape of any numeric
    datatype with elements of the input array replaced according to a map.

    Requires that `id_map_array` contains a mapping for all possible values
    of `input_array`.

    Arguments:
        input_array (array): An nD Numpy array of an unsigned integer type.
        id_map_array (array): A 1D Numpy array with length `k` that serves as
            a map between the values of the input array and the values of the
            returned array. `k` must be the largest possible value for `input_array`.
            The indices of `id_map_array` represent the values of `input_array`
            and the values of `id_map_array` represent the desired values of the
            returned array.

    Returns:
        A Numpy array of the same shape as `input_array` with values according
        to `id_map_array`.
    '''
    return id_map_array[input_array]

def convert_IDs_to_IDs_partial(image, id_map_dict):
    '''
    Converts an array of integers to an array of the same shape of any numeric
    datatype with elements of the input array replaced according to a map.

    This is much slower than `conver_IDs_to_IDs()`, but it doesn't require
    `id_map_dict` to contain a mapping for all possible values of `input_array`,
    but rather only for those values that are to be replaced.

    Arguments:
        input_array (array): An nD Numpy array of an unsigned integer type.
        id_map_dict (array): A Python dictionary that serves as a map between
            the values of the input array and the values of the returned array.
            The keys of `id_map_dict` represent the values of `input_array`
            and the values of `id_map_dict` represent the desired values of the
            returned array.

    Returns:
        A Numpy array of the same shape as `input_array` with values according
        to `id_map_dict`.
    '''
    canvas = np.copy(image)

    for key, value in id_map.items():
        canvas[image == key] = value

    return canvas

def convert_between_IDs_and_colors(image, color_map_dict, gt_dtype=np.uint8):

    if len(np.squeeze(image).shape) == 3:
        canvas = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=gt_dtype)
        for key, value in color_map_dict.items():
            canvas[np.all(image == key, axis=2)] = value
    else:
        canvas = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for key, value in color_map_dict.items():
            canvas[image == key] = value

    return canvas

def convert_IDs_to_colors(image, color_map_array):
    '''
    Converts a array of non-negative integers of shape `(k, ..., m)`
    (such as in a 2D single-channel image with dtype uint8) to an array of shape
    `(k, ..., m, color_array.shape[1])`, with the last axis containing the values
    of `color_map_array`.

    For converting segmentation class IDs to 3-channel colors, this function is
    much faster than `convert_between_IDs_and_colors()`.
    '''

    return color_map_array[image]

def convert_one_hot_to_IDs(one_hot):

    return np.squeeze(np.argmax(one_hot, axis=-1))

def convert_IDs_to_one_hot(image, num_classes):

    unity_vectors = np.eye(num_classes, dtype=np.bool)

    return unity_vectors[image]
