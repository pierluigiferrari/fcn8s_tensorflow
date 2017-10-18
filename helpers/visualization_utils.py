import scipy.misc
import numpy as np

def print_segmentation_onto_image(image, prediction, color_map):
    '''
    Prints a segmentation onto an equally-sized image according to a color map.

    Arguments:
        image (array-like): A 3-channel image onto which to print the segmentation
            from `prediction`.
        prediction (array-like): A rank-4 array that is the segmentation prediction
            with the same spatial dimensions as `image`. The last axis contains the
            segmentation classes in one-hot format.
        color_map (dictionary): A Python dictionary whose keys are non-negative
            integers representing segmentation classes and whose values are 1D tuples
            (or lists, Numpy arrays) of length 4 that represent the RGBA color values
            in which the respective classes are to be annotated. For example, if the
            dictionary contains the key-value pair `{1: (0, 255, 0, 127)}`, then
            this means that all pixels in the prediction that belong to segmentation
            class 1 will be colored in green with 50% transparency in the input image.

    Returns:
        A copy of the input image with the segmentation printed onto it.

    Raises:
        ValueError if the spatial dimensions of `image` and `prediction` don't match.
    '''

    if (image.shape[0] != prediction.shape[1]) or (image.shape[1] != prediction.shape[2]):
        raise ValueError("'image' and 'prediction' must have the same height and width, but image has spatial dimensions ({}, {}) and prediction has spatial dimensions ({}, {}).".format(image.shape[0], image.shape[1], prediction.shape[1], prediction.shape[2]))

    image_size = image.shape

    # Create a template of shape `(image_height, image_width, 4)` to store RGBA values.
    mask = np.zeros(shape=(image_size[0], image_size[1], 4), dtype=np.uint8)
    segmentation_map = np.squeeze(np.argmax(prediction, axis=-1))

    # Loop over all segmentation classes that are to be annotated and put their
    # color value at the respective image pixel.
    for segmentation_class, color_value in color_map.items():

        mask[segmentation_map == segmentation_class] = color_value

    mask = scipy.misc.toimage(mask, mode="RGBA")

    output_image = scipy.misc.toimage(image)
    output_image.paste(mask, box=None, mask=mask) # See http://effbot.org/imagingbook/image.htm#tag-Image.Image.paste for details.

    return output_image
