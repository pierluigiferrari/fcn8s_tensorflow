import numpy as np
import re
import random
import os
import scipy.misc
from glob import glob

def batch_generator(batch_size,
                    dataset_rootdir,
                    images_subdir,
                    labels_subdir,
                    image_size,
                    flip=False):
        """
        Generates batches of iamges and corresponding labels indefinitely.
        Designed for the KITTI Vision Road dataset.

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
