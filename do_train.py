from fcn8s_tensorflow import FCN8s
from data_generator.batch_generator import BatchGenerator
from helpers.visualization_utils import print_segmentation_onto_image, create_video_from_images
from cityscapesscripts.helpers.labels import TRAINIDS_TO_COLORS_DICT, TRAINIDS_TO_RGBA_DICT

from math import ceil
import time
import matplotlib.pyplot as plt

from train_config import train_images, val_images, test_images, train_gt, val_gt
from train_config import num_classes, train_batch_size, val_batch_size
from train_config import vgg_pretrained, epochs

# Put the paths to the datasets in lists, because that's what `BatchGenerator` requires as input.
train_image_dirs = [train_images]
train_ground_truth_dirs = [train_gt]
val_image_dirs = [val_images]
val_ground_truth_dirs = [val_gt]

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

# -----------------------------------------------------------------------------
#                   Dataset Generator
# -----------------------------------------------------------------------------

# Setting same batch size for both generators here.

train_generator = train_dataset.generate(batch_size=train_batch_size,
                                         convert_colors_to_ids=False,
                                         convert_ids_to_ids=False,
                                         convert_to_one_hot=True,
                                         void_class_id=None,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         brightness=False,
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         gray=False,
                                         to_disk=False,
                                         shuffle=True)

val_generator = val_dataset.generate(batch_size=val_batch_size,
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
                                     shuffle=True)

# Print out some diagnostics to make sure that our batches aren't empty and it doesn't take forever to generate them.
start_time = time.time()
images, gt_images = next(train_generator)
print('Time to generate one batch: {:.3f} seconds'.format(time.time() - start_time))
print('Number of images generated:', len(images))
print('Number of ground truth images generated:', len(gt_images))

# Visualize the dataset
# Generate batches from the train_generator where the ground truth does not get converted to one-hot
# so that we can plot it as images.
example_generator = train_dataset.generate(batch_size=train_batch_size,
                                           convert_to_one_hot=False)

# Generate a batch, and visualize.
example_images, example_gt_images = next(example_generator)
i = 0  # Select which sample from the batch to display below.

figure, cells = plt.subplots(1, 2, figsize=(16, 8))
cells[0].imshow(example_images[i])
cells[1].imshow(example_gt_images[i])
plt.show()

# -----------------------------------------------------------------------------
#                   Dataset Generator Ends
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#                   Create the model for Training
# -----------------------------------------------------------------------------
model = FCN8s(model_load_dir=None,
              tags=None,
              vgg16_dir=vgg_pretrained,
              num_classes=num_classes,
              variables_load_dir=None)

# TODO: Define a learning rate schedule function to be passed to the `train()` method.
def learning_rate_schedule(step):
    if step <= 10000:
        return 0.0001
    elif 10000 < step <= 20000:
        return 0.00001
    elif 20000 < step <= 40000:
        return 0.000003
    else:
        return 0.000001


model.train(train_generator=train_generator,
            epochs=epochs,
            steps_per_epoch=ceil(num_train_images / train_batch_size),
            learning_rate_schedule=learning_rate_schedule,
            keep_prob=0.5,
            l2_regularization=0.0,
            eval_dataset='val',
            eval_frequency=2,
            val_generator=val_generator,
            val_steps=ceil(num_val_images / val_batch_size),
            metrics={'loss', 'mean_iou', 'accuracy'},
            save_during_training=True,
            save_dir='cityscapes_model',
            save_best_only=True,
            save_tags=['default'],
            save_name='(batch-size-4)',
            save_frequency=2,
            saver='saved_model',
            monitor='loss',
            record_summaries=True,
            summaries_frequency=10,
            summaries_dir='tensorboard_log\\cityscapes',
            summaries_name='configuration_01',
            training_loss_display_averaging=3)

model.save(model_save_dir='cityscapes_model',
           saver='saved_model',
           tags=['default'],
           name='(batch-size-4)',
           include_global_step=True,
           include_last_training_loss=True,
           include_metrics=True,
           force_save=False)


model.evaluate(data_generator=val_generator,
               metrics={'loss', 'mean_iou', 'accuracy'},
               num_batches=ceil(num_val_images/val_batch_size),
               l2_regularization=0.0,
               dataset='val')