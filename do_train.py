from fcn8s_tensorflow import FCN8s
from data_generator.batch_generator import BatchGenerator
from helpers.visualization_utils import print_segmentation_onto_image, create_video_from_images
from cityscapesscripts.helpers.labels import TRAINIDS_TO_COLORS_DICT, TRAINIDS_TO_RGBA_DICT

from math import ceil
import time
import matplotlib.pyplot as plt

# TODO: Set the paths to the images.
train_images = '/home/codesteller/datasets/cityscapes/cityscapes/leftImg8bit/train/'
val_images = '/home/codesteller/datasets/cityscapes/cityscapes/leftImg8bit/val/'
test_images = '/home/codesteller/datasets/cityscapes/cityscapes/leftImg8bit/test/'

# TODO: Set the paths to the ground truth images.
train_gt = '/home/codesteller/datasets/cityscapes/cityscapes/gtFine/train/'
val_gt = '/home/codesteller/datasets/cityscapes/cityscapes/gtFine/val/'

# Put the paths to the datasets in lists, because that's what `BatchGenerator` requires as input.
train_image_dirs = [train_images]
train_ground_truth_dirs = [train_gt]
val_image_dirs = [val_images]
val_ground_truth_dirs = [val_gt]

num_classes = 20  # TODO: Set the number of segmentation classes.

train_dataset = BatchGenerator(image_dirs=train_image_dirs,
                               image_file_extension='png',
                               ground_truth_dirs=train_ground_truth_dirs,
                               image_name_split_separator='leftImg8bit',
                               ground_truth_suffix='gtFine_labelIds',
                               check_existence=True,
                               num_classes=num_classes)

val_dataset = BatchGenerator(image_dirs=val_image_dirs,
                             image_file_extension='png',
                             ground_truth_dirs=val_ground_truth_dirs,
                             image_name_split_separator='leftImg8bit',
                             ground_truth_suffix='gtFine_labelIds',
                             check_existence=True,
                             num_classes=num_classes)

num_train_images = train_dataset.get_num_files()
num_val_images = val_dataset.get_num_files()

print("Size of training dataset: ", num_train_images, " images")
print("Size of validation dataset: ", num_val_images, " images")
