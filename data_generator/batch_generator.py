import numpy as np
import random
import os
import sys
import pathlib
import imageio
import cv2
from glob import glob
from math import ceil
from tqdm import trange

from helpers.ground_truth_conversion_utils import convert_IDs_to_IDs, convert_IDs_to_one_hot, \
    convert_between_IDs_and_colors, convert_IDs_to_IDs_partial


class BatchGenerator():

    def __init__(self,
                 image_dirs,
                 image_file_extension='png',
                 ground_truth_dirs=None,
                 image_name_split_separator=None,
                 ground_truth_suffix=None,
                 check_existence=True,
                 num_classes=None,
                 root_dir=None,
                 export_dir=None):
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
            ground_truth_dirs (list, optional): `None` or a list of directory paths,
                each of which contain the ground truth images that correspond to
                the respective directory paths in `datasets`. The ground truth
                images must have 1 channel that encodes the segmentation classes
                numbered consecutively from 0 to `n`, where `n` is an integer.
            image_name_split_separator (string, optional): Only relevant if
                `ground_truth_dirs` contains at least one item. A string by which
                the image names will be split into a left and right part, the left
                part of which (i.e. the beginning of the image file name) will be
                used to get the matching ground truth image file name. More precisely,
                all characters left of the separator string will constitute the
                beginning of the file name of the corresponding ground truth image.
            ground_truth_suffix (string, optional): The suffix added to the left part of
                an image name string (see `image_name_split_separator`) in order
                to compose the name of the corresponding ground truth image file.
                The suffix must exclude the file extension.
            check_existence (bool, optional): Only relevant if ground truth images
                are given. If `True`, the constructor checks for each ground truth image
                path whether the respective file actually exists and throws a
                `DataError` if it doesn't. Defaults to `True`.
            num_classes (int, optional): The number of segmentation classes in the
                ground truth data. Only relevant if you want the generator to convert
                numeric labels to one-hot format, otherwise you can leave this `None`.
            root_dir (string, optional): The dataset root directory. This is only
                relevant if you want to use the generator to save processed data
                to disk in addition to yielding it, i.e. if you want to do offline processing.
                In this case, the generator will reproduce the directory hierarchy
                of the source data within the target directory in which to save
                the processed data. It needs to know the root directory of the
                dataset in order to do so
            export_dir (string, optional): This is only relevant if you want use
                the generator to save processed data to disk in addition to yielding it,
                i.e. if you want to do offline processing. This is the directory
                into which the processed data will be written. The generator will
                reproduce the directory hierarchy of the source data within this
                directory.
        '''

        self.image_dirs = image_dirs
        self.ground_truth_dirs = ground_truth_dirs
        self.root_dir = root_dir # The dataset root directory.
        self.export_dir = export_dir
        self.image_paths = [] # The list of images from which the generator will draw.
        self.ground_truth_paths = {} # The dictionary of ground truth images that correspond to the images.
        self.num_classes = num_classes
        self.dataset_size = 0
        self.ground_truth = False # Whether or not ground truth images were given.

        if (not self.ground_truth_dirs is None) and (len(self.image_dirs) != len(self.ground_truth_dirs)):
            raise ValueError("`image_dirs` and `ground_truth_dirs` must contain the same number of elements.")

        image_file_extension = image_file_extension.lower()

        for i, image_dir in enumerate(image_dirs): # Iterate over all given datasets.

            for image_dir_path, subdir_list, file_list in os.walk(image_dir, topdown=True): # Iterate over all subdirectories of this dataset directory.

                image_paths = glob(os.path.join(image_dir_path, '*.' + image_file_extension)) # Get all images in this directory

                if len(image_paths) > 0: # If there are any images, add them to the list of images.

                    self.image_paths += image_paths

                    if not ground_truth_dirs is None: # If there is ground truth data, add it to the ground truth list.
                        # Get the path of the ground truth directory that corresponds to this image directory.
                        ground_truth_dir = ground_truth_dirs[i] # Get the head.
                        ground_truth_subdir = os.path.basename(os.path.normpath(image_dir_path)) # Get the subdirectory we're currently in.
                        ground_truth_dir_path = os.path.join(ground_truth_dir, ground_truth_subdir)

                        # Loop over all image paths to collect the corresponding ground truth image paths.
                        for image_path in image_paths:
                            # Construct the name of the ground truth image from the name of the image.
                            image_name = os.path.basename(image_path)
                            left_part = image_name.split(image_name_split_separator, 1)[0] # Get the left part of the split.
                            # Compose the name of the ground truth image that corresponds to this image.
                            ground_truth_image_name = left_part + ground_truth_suffix + '.' + image_file_extension
                            # Create the full path to this ground truth image.
                            ground_truth_path = os.path.join(ground_truth_dir_path, ground_truth_image_name)

                            if check_existence and not os.path.isfile(ground_truth_path):
                                raise DataError("The dataset contains an image file '{}' for which the corresponding ground truth image file does not exist at '{}'.".format(image_path, ground_truth_path))

                            # Add the pair `image_name : ground_truth_path` to the dictionary
                            self.ground_truth_paths[image_name] = ground_truth_path

        self.dataset_size = len(self.image_paths)

        if self.dataset_size == 0:
            raise DataError("No images with the given file extension '{}' were found in the given image directories.".format(image_file_extension))

        if (not ground_truth_dirs is None) and (len(self.ground_truth_paths) != self.dataset_size):
            raise DataError('Ground truth directories were given, but the number of ground truth images found does not match the number of images. Number of images: {}. Number of ground truth images: {}'.format(self.dataset_size, len(self.ground_truth_paths)))

        if len(self.ground_truth_paths) > 0:
            self.ground_truth = True

    def get_num_files(self):
        '''
        Returns the total number of image files (or image/ground truth image file
        pairs if ground truth data was given) contained in all dataset directories
        passed to the BatchGenerator constructor.
        '''
        return self.dataset_size

    def generate(self,
                 batch_size,
                 convert_colors_to_ids=False,
                 convert_ids_to_ids=False,
                 convert_to_one_hot=True,
                 void_class_id=None,
                 random_crop=False,
                 crop=False,
                 resize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 gray=False,
                 to_disk=False,
                 shuffle=True):
        '''

        With any of the image transformations below, the respective ground truth images, if given,
        will be transformed accordingly.

        Arguments:
            batch_size (int): The number of images (or image/ground truth pairs) to generate per
                batch.
            convert_colors_to_ids (dict, optional): `False` or a dictionary in which the keys are
                3-tuples of `dtype uint8` representing 3-channel color values and the values
                are integers representing the segmentation class ID associated with a given color
                value. If the input ground truth images are 3-channel color images and a conversion
                dictionary is passed, the ground truth images will be converted to single-channel
                images with the according class IDs instead of color values. It is recommended to
                perform color-to-ID conversion offline.
            convert_ids_to_ids (array or dict, optional): `False` or either a 1D Numpy array or a Python
                dictionary that represents a map according to which the generator will convert the
                grund truth data's current class IDs to the desired class IDs. In the case of an array,
                the array's indices represent the current IDs and the array's integer values represent the
                desired IDs to which to convert. The array must contain a map for all possible unique
                current class IDs. In the case of a dictionary, both keys and values must be integers.
                The keys are the current IDs and the values are the desired IDs to which to convert.
                The dictionary does not need to contain a mapping for all possible unique current class IDs.
                For conversion of all IDs, an array will enable much faster conversion than a dictionary.
            convert_to_one_hot (bool, optional): If `True`, the ground truth data will be converted to
                one-hot format.
            void_class_id (int, optional): The class ID of a 'void' or 'background' class. Only relevant
                if any of the `random_crop`, `translate`, or `scale` transformations are being used
                on ground truth data. Determines the pixel value of blank image space that might occur
                through the aforementioned transformations.
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
            to_disk (bool, optional): If `True`, the generated batches are being saved to `export_dir` (see constuctor)
                in addition to being yielded. This can be used for offline dataset processing.
            shuffle (bool, optional): If `True`, the dataset will be shuffled before each new pass.

        Yields:
            Either one 4D Numpy array of shape `(batch_size, img_height, img_with, num_channels)` with the
            generated images, or, if paths to ground truth data were passed in the constructor, two Numpy
            arrays, the first is the same as in the former case and the second has shape
            `(batch_size, img_height, img_with)` and contains the generated ground truth images.
        '''
        if (convert_to_one_hot or (not convert_colors_to_ids is False) or (not convert_ids_to_ids is False)) and not self.ground_truth:
            raise ValueError("Cannot convert ground truth data: No ground truth data given.")

        if convert_to_one_hot and self.num_classes is None:
            raise ValueError("One-hot conversion requires that you pass an integer value for `num_classes` in the constructor, but `num_classes` is `None`.")

        if shuffle:
            random.shuffle(self.image_paths)

        current = 0

        while True:

            # Store the new batch here
            images = []
            gt_images = []

            # Shuffle data after each complete pass
            if current >= len(self.image_paths):
                if shuffle: random.shuffle(self.image_paths)
                current = 0

            # Load the images and ground truth images for this batch
            for image_path in self.image_paths[current:current+batch_size]: # Careful: This works in Python, but might cause an 'index out of bounds' error in other languages if `current+batch_size > len(image_paths)`

                # Load the image
                image = imageio.imread(image_path)
                img_height, img_width, img_ch = image.shape

                # If at least one ground truth directory was given, load the ground truth images.
                if self.ground_truth:

                    gt_image_path = self.ground_truth_paths[os.path.basename(image_path)]
                    gt_image = imageio.imread(gt_image_path)
                    gt_dtype = gt_image.dtype

                    if not convert_colors_to_ids is False:
                        gt_image = convert_between_IDs_and_colors(gt_image, convert_colors_to_ids, gt_dtype=gt_dtype)

                    if not convert_ids_to_ids is False:
                        if isinstance(convert_ids_to_ids, np.ndarray):
                            gt_image = convert_IDs_to_IDs(gt_image, convert_ids_to_ids)
                        if isinstance(convert_ids_to_ids, dict):
                            gt_image = convert_IDs_to_IDs_partial(gt_image, convert_ids_to_ids)

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
                            canvas = np.full(shape=random_crop, fill_value=void_class_id, dtype=gt_dtype) # ...generate a blank background image to place the patch onto,...
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
                            canvas = np.full(shape=random_crop, fill_value=void_class_id, dtype=gt_dtype) # ...generate a blank background image to place the patch onto,...
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
                            canvas = np.full(shape=random_crop, fill_value=void_class_id, dtype=gt_dtype) # ...generate a blank background image to place the patch onto,...
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
                        if self.ground_truth: gt_image = cv2.warpAffine(src=gt_image, M=translation_matrix, dsize=(img_width, img_height), borderValue=void_class_id)

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
                                canvas = np.full(shape=(img_height, img_width), fill_value=void_class_id, dtype=gt_dtype)
                                canvas[y_offset:y_offset+scaled_height, x_offset:x_offset+scaled_width] = patch_gt_image
                                gt_image = canvas
                            if scaling_factor > 1:
                                gt_image = np.copy(patch_gt_image[y_offset:img_height+y_offset, x_offset:img_width+x_offset])

                if gray:
                    image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=2)

                # Maybe convert ground truth IDs to one-hot.
                if convert_to_one_hot:
                    gt_image = convert_IDs_to_one_hot(gt_image, self.num_classes)

                if to_disk: # If the processed data is to be written to disk instead of yieled.
                    # Create the directory (including parents) if it doesn't already exist.
                    image_save_file_path = os.path.join(self.export_dir, os.path.relpath(image_path, start=self.root_dir))
                    image_save_directory_path = os.path.dirname(image_save_file_path)
                    pathlib.Path(image_save_directory_path).mkdir(parents=True, exist_ok=True)
                    # Save the image.
                    imageio.imsave(image_save_file_path, image)
                    if self.ground_truth:
                        # Create the directory (including parents) if it doesn't already exist.
                        gt_image_save_file_path = os.path.join(self.export_dir, os.path.relpath(gt_image_path, start=self.root_dir))
                        gt_image_save_directory_path = os.path.dirname(gt_image_save_file_path)
                        pathlib.Path(gt_image_save_directory_path).mkdir(parents=True, exist_ok=True)
                        # Save the ground truth image.
                        imageio.imsave(gt_image_save_file_path, gt_image)

                # Append the processed image (and maybe ground truth image) to this batch.
                images.append(image)
                if self.ground_truth: gt_images.append(gt_image)

            current += batch_size

            if self.ground_truth:
                yield np.array(images), np.array(gt_images)
            else:
                yield np.array(images)

    def process_all(self,
                    convert_colors_to_ids=False,
                    convert_ids_to_ids=False,
                    convert_to_one_hot=False,
                    void_class_id=None,
                    random_crop=False,
                    crop=False,
                    resize=False,
                    brightness=False,
                    flip=False,
                    translate=False,
                    scale=False,
                    gray=False,
                    to_disk=True,
                    shuffle=False,
                    batch_size=1):
        '''
        Processes the entire dataset in batches of `batch_size` and saves the
        results to `export_dir` (see constructor).

        This is basically just a wrapper around the `generate()` method that
        iterates over the entire dataset (or datasets, in case multiple were
        passed in the constructor).

        For documentation of the arguments, see `generate()`. Returns void.
        '''

        preprocessor = self.generate(batch_size=batch_size,
                                convert_colors_to_ids=convert_colors_to_ids,
                                convert_ids_to_ids=convert_ids_to_ids,
                                convert_to_one_hot=convert_to_one_hot,
                                void_class_id=void_class_id,
                                random_crop=random_crop,
                                crop=crop,
                                resize=resize,
                                brightness=brightness,
                                flip=flip,
                                translate=translate,
                                scale=scale,
                                gray=gray,
                                to_disk=to_disk,
                                shuffle=shuffle)

        num_batches = ceil(self.dataset_size/batch_size)

        tr = trange(num_batches, file=sys.stdout)
        tr.set_description('Processing images')

        for batch in tr:
                next(preprocessor)


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

class DataError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
