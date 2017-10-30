import scipy.misc
import numpy as np
from moviepy.editor import ImageSequenceClip
from glob import glob
import os

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

def create_split_view(target_size, images, positions, sizes, captions=[]):
    '''
    Place images onto a rectangular canvas to create a split view.

    Arguments:
        target_size (tuple): The target size of the output canvas in the format
            (height, width). The output canvas will always have three color channels.
        images (list): A list containing the images to be placed onto the output canvas.
            The images can vary in size and can have one or three color channels.
        positions (list): A list containing the desired top left corner positions of
            the images in the output canvas in the format (y, x), where x refers
            to the horizontal coordinate and y refers to the vertical coordinate
            and both are non-negative integers.
        sizes (list): A list containing tuples with the desired sizes of the images
            in the format (height, width).
        captions (list, optional): A list containing either a caption string or
            `None` for each image. The list must have the same length as `images`.
            Defaults to an empty list, i.e. no captions will be added.

    Returns:
        The split view image of size `target_size`.
    '''

    assert len(images) == len(positions) == len(sizes), "`images`, `positions`, and `sizes` must have the same length, but it is `len(images) == {}`, `len(poisitons) = {}`, `len(sizes) == {}`".format(len(images), len(positions), len(sizes))

    y_max, x_max = target_size
    canvas = np.zeros((y_max, x_max, 3), dtype=np.uint8)

    for i, img in enumerate(images):

        # Resize the image
        if img.shape[0] != sizes[i][0] | img.shape[1] != sizes[i][1]:
            img = scipy.misc.imresize(img, sizes[i])

        # Place the resized image onto the canvas
        y, x = positions[i]
        h, w = sizes[i]
        # If img is grayscale, Numpy broadcasting will put the same intensity value for each the R, G, and B channels.
        # The indexing below protects against index-out-of-range issues.
        canvas[y:min(y + h, y_max), x:min(x + w, x_max), :] = img[:min(h, y_max - y), :min(w, x_max - x)]

        # Print captions onto the canvas if there are any
        if captions and (captions[i] is not None):
            cv2.putText(canvas, "{}".format(captions[i]), (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    return canvas

def create_video_from_images(video_output_name, image_input_dir, frame_rate=30.0, image_file_extension='png'):
    '''
    Creates an MP4 video from the images in a given directory.

    Arguments:
        video_output_name (string): The full path and name of the output video
            excluding the file extension. The output video will be in MP4 format.
        image_input_dir (string): The directory that contains the input images.
        frame_rate (float, optional): The number of frames per second.
        image_file_extension: The file extension of the source images. Only images
            with a matching file extension will be included in the video.
            Defaults to 'png'.
    '''

    image_paths = glob(os.path.join(image_input_dir, '*.' + image_file_extension))
    image_paths = sorted(image_paths)

    video = ImageSequenceClip(image_paths, fps=frame_rate)
    video.write_videofile("{}.mp4".format(video_output_name))
