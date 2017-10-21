import numpy as np
import re
import random
import os
import scipy.misc
import cv2
from glob import glob

def batch_generator(batch_size,
                    dataset_rootdir,
                    images_subdir,
                    labels_subdir,
                    image_size,
                    flip=False):
        """
        Generates batches of iamges and corresponding labels indefinitely.

        Returns a tuple of two Numpy arrays, one containing the next `batch_size`
        images, the other containing the corresponding labels.

        Shuffles images and labels consistently after each complete pass.

        Arguments:
            batch_size (int): The number of samples per batch.
            dataset_rootdir (string): The root directory of the dataset. Both the
                images and the labels must be located in some sub-directory of
                this directory. Trailing slashes do not matter.
            images_subdir (string): The sub-directory within the dataset root
                directory within which the images are located. Leading or trailing
                slashes do not matter.
            labels_subdir (string): `None` or the sub-directory within the dataset
                root directory in which the labels are located. Leading or
                trailing slashes do not matter.
            image_size (tuple): A tuple that represents the size to which all
                images will be resized in the format `(height, width)`.

        """

        image_paths = glob(os.path.join(dataset_rootdir, images_subdir, '*.png'))
        if not labels_subdir is None:
            label_paths = {re.sub(r'_road_', '_', os.path.basename(path)): path
                           for path in glob(os.path.join(dataset_rootdir, labels_subdir, '*_road_*.png'))}

        # The background color for the labels in the KITTI road dataset.
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)

        current = 0

        while True:

            # Store the new batch here
            images = []
            labels = []

            # Shuffle data after each complete pass
            if current >= len(image_paths):
                random.shuffle(image_paths)
                current = 0

            # Load the images and labels for this batch
            for image_path in image_paths[current:current+batch_size]: # Careful: This works in Python, but might cause an 'index out of bounds' error in other languages if `current+batch_size > len(image_paths)`

                # Load the images

                # TODO: Images shouldn't be resized here. This is just a quick and
                #       dirty solution until batch generation and image manipulation
                #       will be separated properly (and random crops are a lot better
                #       in many cases than resizing).
                image = scipy.misc.imresize(scipy.misc.imread(image_path), image_size)
                images.append(image)

                # If a label path was given, load the labels
                if not labels_subdir is None:

                    label_path = label_paths[os.path.basename(image_path)]
                    label = scipy.misc.imresize(scipy.misc.imread(label_path), image_size)

                    # Process the labels:
                    # Convert the RGB label images to boolean arrays where background = false and road = true.
                    label_background = np.all(label == background_color, axis=2) # Array of shape (height, width) where every background pixel is `True`.
                    label_background = np.expand_dims(label_background, -1) # Array of shape (height, width, 1).
                    label = np.concatenate((label_background, np.invert(label_background)), axis=2) # Array of shap (height, width, 2) where the first channel is for the background pixels and the second for the road pixels.

                    labels.append(label)

            current += batch_size

            # At this point we're done producing the batch. Now perform some
            # optional image transformations:

            for i in range(len(images)):

                img_height, img_width, ch = images[i].shape # Get the dimensions of this image.

                if flip:
                    p = np.random.uniform(0,1)
                    if p >= (1-flip):
                        images[i] = images[i][:,::-1,:]
                        if not labels_subdir is None:
                            labels[i] = labels[i][:,::-1,:]

            if not labels_subdir is None:
                yield np.array(images), np.array(labels)
            else:
                yield np.array(images)

class BatchGenerator():

    def __init__(self,
                 image_dirs,
                 image_file_extension='png',
                 ground_truth_dirs=[],
                 image_name_split_separator=None,
                 ground_truth_suffix=None):
        '''
        Arguments:
            image_dirs (list): A list of directory paths, each of which contain
                images either directly or within a hierarchy of subdirectories.
                The directory paths given serve as root directories and the generator
                will load images from all subdirectories within. This lets you
                combine multiple datasets randomly. All images must have 3 channels.
            image_file_extension (string, optional): The file extension of the
                images in the datasets. Must be identical for all images in all
                datasets in `datasets`. Defaults to `png`.
            ground_truth_dirs (list, optional): An optional list of directory paths,
                each of which contain the ground truth images that correspond to
                the respective directory paths in `datasets`. The ground truth
                images must have 1 channel that encodes the segmentation classes
                numbered consecutively from 0 to `n`, where `n` is an integer.
                It is possible to define a list of void/background class labels
                that will all be mapped to one user-definable background class ID.
                This might be relevant for datasets where some void classes are
                labeled with -1 or 255. In any case, all class labels must end up
                being consecutive integers, starting at 0. This list can be empty,
                in which case the generator will yield only images, no ground
                truth images. The length of this list must be either zero or
                the same as the length of `image_dirs`.
            image_name_split_separator (string, optional): Only relevant if
                `ground_truth_dirs` contains at least one item. A string by which
                the image names will be split into a left and right part, the left
                part of which (i.e. the beginning of the image file name) will be
                used to get the matching ground truth image file name. More precisely,
                all characters left of the separator string will constitute the
                beginning of the file name of the corresponding ground truth image.
            ground_truth_suffix (string): The suffix added to the left part of
                an image name string (see `image_name_split_separator`) in order
                to compose the name of the corresponding ground truth image file.
                The suffix must exclude the file extension.
        '''

        self.image_paths = [] # The list of images from which the generator will draw.
        self.ground_truth_paths = {} # The dictionary of ground truth images that correspond to the images.
        self.dataset_size = 0
        self.ground_truth = False # Whether or not ground truth images were given.

        image_file_extension = image_file_extension.lower()

        for i, image_dir in enumerate(image_dirs): # Iterate over all given datasets.

            for image_dir_path, subdir_list, file_list in os.walk(image_dir, topdown=True): # Iterate over all subdirectories of this dataset directory.

                image_paths = glob(os.path.join(image_dir_path, '*.' + image_file_extension)) # Get all images in this directory

                if len(image_paths) > 0: # If there are any images, add them to the list of images.

                    self.image_paths += image_paths

                    if len(ground_truth_dirs) > 0: # If there is ground truth data, add it to the ground truth list.
                        # Get the path of the ground truth directory that corresponds to this image directory.
                        ground_truth_dir = ground_truth_dirs[i] # Get the head.
                        ground_truth_subdir = os.path.basename(os.path.normpath(image_dir_path)) # Get the subdirectory we're currently in.
                        ground_truth_dir_path = os.path.join(ground_truth_dir, ground_truth_subdir)

                        for image_path in image_paths:
                            # Construct the name of the ground truth image from the name of the image.
                            image_name = os.path.basename(image_path)
                            left_part = image_name.split(image_name_split_separator, 1)[0] # Get the left part of the split.
                            ground_truth_image_name = left_part + ground_truth_suffix + '.' + image_file_extension

                            # Add the pair `image_name : ground_truth_path` to the dictionary
                            self.ground_truth_paths[image_name] = os.path.join(ground_truth_dir_path, ground_truth_image_name)

        self.dataset_size = len(self.image_paths)

        if self.dataset_size == 0:
            raise SizeError("No images with the given file extension '{}' were found in the given image directories.".format(image_file_extension))

        if (len(ground_truth_dirs) > 0) and (len(self.ground_truth_paths) != self.dataset_size):
            raise SizeError('Ground truth directories were given, but the number of ground truth images found does not match the number of images. Number of images: {}. Number of ground truth images: {}'.format(self.dataset_size, len(self.ground_truth_paths)))

        if len(self.ground_truth_paths) > 0:
            self.ground_truth = True

    def generate(self,
                 batch_size,
                 void_classes={},
                 void_class_ID=None,
                 conver_IDs_to_one_hot=True,
                 random_crop=False,
                 crop=False,
                 resize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 gray=False):
        '''

        With any of the image transformations below, the respective ground truth images, if given,
        will be transformed accordingly.

        Arguments:
            random_crop (tuple, optional): `False` or a tuple of two integers, `(height, width)`,
                where `height` and `width` are the height and width of the patch that is to be cropped out at a random
                position in the input image. Note that `height` and `width` can be arbitrary - they are allowed to be larger
                than the image height and width, in which case the original image will be randomly placed on a black background
                canvas of size `(height, width)`. Defaults to `False`.
            crop (tuple, optional): `False` or a tuple of four integers, `(crop_top, crop_bottom, crop_left, crop_right)`,
                with the number of pixels to crop off of each side of the images. Note: Cropping happens after random cropping.
            resize (tuple, optional): `False` or a tuple of 2 integers for the desired output
                size of the images in pixels. The expected format is `(height, width)`.
                Note: Resizing happens after both random cropping and cropping.
            brightness (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the brightness of the image by a factor randomly picked from a uniform
                distribution in the boundaries of `[min, max]`. Both min and max must be >=0.
            flip (float, optional): `False` or a float in [0,1], see `prob` above. Flip the image horizontally.
            translate (tuple, optional): `False` or a tuple, with the first two elements tuples containing
                two integers each, and the third element a float: `((min, max), (min, max), prob)`.
                The first tuple provides the range in pixels for the horizontal shift of the image,
                the second tuple for the vertical shift. The number of pixels to shift the image
                by is uniformly distributed within the boundaries of `[min, max]`, i.e. `min` is the number
                of pixels by which the image is translated at least. Both `min` and `max` must be >=0.
            scale (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the image by a factor randomly picked from a uniform distribution in the boundaries
                of `[min, max]`. Both `min` and `max` must be >=0.
            gray (bool, optional): If `True`, converts the images to grayscale. Note that the resulting grayscale
                images have shape `(height, width, 1)`.
        '''

        random.shuffle(self.image_paths)

        current = 0

        while True:

            # Store the new batch here
            images = []
            gt_images = []

            # Shuffle data after each complete pass
            if current >= len(self.image_paths):
                random.shuffle(self.image_paths)
                current = 0

            # Load the images and ground truth images for this batch
            for image_path in self.image_paths[current:current+batch_size]: # Careful: This works in Python, but might cause an 'index out of bounds' error in other languages if `current+batch_size > len(image_paths)`

                # Load the image
                image = scipy.misc.imread(image_path)
                img_height, img_width, img_ch = image.shape

                # If at least one ground truth directory was given, load the ground truth images.
                if self.ground_truth:

                    gt_image = scipy.misc.imread(self.ground_truth_paths[os.path.basename(image_path)])

                # Maybe process the images and ground truth images.

                if random_crop:
                    # Compute how much room we have in both dimensions to make a random crop.
                    # A negative number here means that we want to crop out a patch that is larger than the original image in the respective dimension,
                    # in which case we will create a black background canvas onto which we will randomly place the image.
                    y_range = img_height - random_crop[0]
                    x_range = img_width - random_crop[1]

                    # Select a random crop position from the possible crop positions
                    if y_range >= 0: crop_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension
                    else: crop_ymin = np.random.randint(0, -y_range + 1) # The possible positions for the image on the background canvas in the vertical dimension
                    if x_range >= 0: crop_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension
                    else: crop_xmin = np.random.randint(0, -x_range + 1) # The possible positions for the image on the background canvas in the horizontal dimension
                    # Perform the crop
                    if y_range >= 0 and x_range >= 0: # If the patch to be cropped out is smaller than the original image in both dimenstions, we just perform a regular crop
                        # Crop the image
                        image = np.copy(image[crop_ymin:crop_ymin+random_crop[0], crop_xmin:crop_xmin+random_crop[1]])
                        # Do the same for the ground truth image.
                        if self.ground_truth: gt_image = np.copy(gt_image[crop_ymin:crop_ymin+random_crop[0], crop_xmin:crop_xmin+random_crop[1]])
                    elif y_range >= 0 and x_range < 0: # If the crop is larger than the original image in the horizontal dimension only,...
                        # Crop the image
                        patch_image = np.copy(image[crop_ymin:crop_ymin+random_crop[0]]) # ...crop the vertical dimension just as before,...
                        canvas = np.zeros(shape=(random_crop[0], random_crop[1], patch_image.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                        canvas[:, crop_xmin:crop_xmin+img_width] = patch_image # ...and place the patch onto the canvas at the random `crop_xmin` position computed above.
                        image = canvas
                        # Do the same for the ground truth image.
                        if self.ground_truth:
                            patch_gt_image = np.copy(gt_image[crop_ymin:crop_ymin+random_crop[0]]) # ...crop the vertical dimension just as before,...
                            canvas = np.full(shape=random_crop, fill_value=void_class_ID, dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                            canvas[:, crop_xmin:crop_xmin+img_width] = patch_gt_image # ...and place the patch onto the canvas at the random `crop_xmin` position computed above.
                            gt_image = canvas
                    elif y_range < 0 and x_range >= 0: # If the crop is larger than the original image in the vertical dimension only,...
                        # Crop the image
                        patch_image = np.copy(image[:,crop_xmin:crop_xmin+random_crop[1]]) # ...crop the horizontal dimension just as in the first case,...
                        canvas = np.zeros(shape=(random_crop[0], random_crop[1], patch_image.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                        canvas[crop_ymin:crop_ymin+img_height, :] = patch_image # ...and place the patch onto the canvas at the random `crop_ymin` position computed above.
                        image = canvas
                        # Do the same for the ground truth image.
                        if self.ground_truth:
                            patch_gt_image = np.copy(gt_image[:,crop_xmin:crop_xmin+random_crop[1]]) # ...crop the horizontal dimension just as in the first case,...
                            canvas = np.full(shape=random_crop, fill_value=void_class_ID, dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                            canvas[crop_ymin:crop_ymin+img_height, :] = patch_gt_image # ...and place the patch onto the canvas at the random `crop_ymin` position computed above.
                            gt_image = canvas
                    else:  # If the crop is larger than the original image in both dimensions,...
                        patch_image = np.copy(image)
                        canvas = np.zeros(shape=(random_crop[0], random_crop[1], patch_image.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                        canvas[crop_ymin:crop_ymin+img_height, crop_xmin:crop_xmin+img_width] = patch_image # ...and place the patch onto the canvas at the random `(crop_ymin, crop_xmin)` position computed above.
                        image = canvas
                        # Do the same for the ground truth image.
                        if self.ground_truth:
                            patch_gt_image = np.copy(gt_image)
                            canvas = np.full(shape=random_crop, fill_value=void_class_ID, dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                            canvas[crop_ymin:crop_ymin+img_height, crop_xmin:crop_xmin+img_width] = patch_gt_image # ...and place the patch onto the canvas at the random `(crop_ymin, crop_xmin)` position computed above.
                            gt_image = canvas
                    # Update the height and width values.
                    img_height, img_width = random_crop

                if crop:
                    image = np.copy(image[crop[0]:img_height-crop[1], crop[2]:img_width-crop[3]])
                    gt_image = np.copy(gt_image[crop[0]:img_height-crop[1], crop[2]:img_width-crop[3]])

                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
                    if self.ground_truth: gt_image = cv2.resize(gt_image, dsize=(resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
                    img_height, img_width = resize # Updating these at this point is unnecessary, but it's one fewer source of error if this method gets expanded in the future

                if brightness:
                    p = np.random.uniform(0,1)
                    if p >= (1-brightness[2]):
                        image = _brightness(image, min=brightness[0], max=brightness[1])

                if flip:
                    p = np.random.uniform(0,1)
                    if p >= (1-flip):
                        image = cv2.flip(image, 1) # Horizontal flip
                        if self.ground_truth: gt_image = cv2.flip(gt_image, 1) # Horizontal flip

                if translate:
                    p = np.random.uniform(0,1)
                    if p >= (1-translate[2]):
                        # Randomly select horizontal and vertical shift values.
                        x = np.random.randint(translate[0][0], translate[0][1]+1)
                        y = np.random.randint(translate[1][0], translate[1][1]+1)
                        x_shift = random.choice([-x, x])
                        y_shift = random.choice([-y, y])
                        # Compute the warping matrix for the selected values.
                        translation_matrix = np.float32([[1,0,x_shift],[0,1,y_shift]])
                        # Warp the image and maybe the ground truth image.
                        image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(img_width, img_height))
                        if self.ground_truth: gt_image = cv2.warpAffine(src=gt_image, M=translation_matrix, dsize=(img_width, img_height), borderValue=void_class_ID)

                if scale:
                    p = np.random.uniform(0,1)
                    if p >= (1-scale[2]):
                        scaling_factor = np.random.uniform(scale[0], scale[1])
                        scaled_height = int(img_height * scaling_factor)
                        scaled_width = int(img_width * scaling_factor)
                        y_offset = abs(int((img_height - scaled_height) / 2))
                        x_offset = abs(int((img_width - scaled_width) / 2))

                        # Scale the image.
                        patch_image = cv2.resize(image, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
                        if scaling_factor <= 1:
                            canvas = np.zeros(shape=(img_height, img_width, img_ch), dtype=np.uint8)
                            canvas[y_offset:y_offset+scaled_height, x_offset:x_offset+scaled_width] = patch_image
                            image = canvas
                        if scaling_factor > 1:
                            image = np.copy(patch_image[y_offset:img_height+y_offset, x_offset:img_width+x_offset])

                        # Scale the ground truth image.
                        if self.ground_truth:
                            patch_gt_image = cv2.resize(gt_image, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
                            if scaling_factor <= 1:
                                canvas = np.full(shape=(img_height, img_width), fill_value=void_class_ID, dtype=np.uint8)
                                canvas[y_offset:y_offset+scaled_height, x_offset:x_offset+scaled_width] = patch_gt_image
                                gt_image = canvas
                            if scaling_factor > 1:
                                gt_image = np.copy(patch_gt_image[y_offset:img_height+y_offset, x_offset:img_width+x_offset])

                if gray:
                    image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=2)

                # Append the processed image (and maybe ground truth image) to this batch.
                images.append(image)
                if self.ground_truth: gt_images.append(gt_image)

            current += batch_size

            if self.ground_truth:
                yield np.array(images), np.array(gt_images)
            else:
                yield np.array(images)

def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly changes the brightness of the input image.

    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min,max)

    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel

    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

class SizeError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
