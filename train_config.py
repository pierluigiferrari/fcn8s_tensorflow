# TODO: Set the paths to the images.
train_images = '/home/codesteller/datasets/cityscapes/cityscapes/leftImg8bit/train/'
val_images = '/home/codesteller/datasets/cityscapes/cityscapes/leftImg8bit/val/'
test_images = '/home/codesteller/datasets/cityscapes/cityscapes/leftImg8bit/test/'

# TODO: Set the paths to the ground truth images.
train_gt = '/home/codesteller/datasets/cityscapes/cityscapes/gtFine/train/'
val_gt = '/home/codesteller/datasets/cityscapes/cityscapes/gtFine/val/'

num_classes = 34  # TODO: Set the number of segmentation classes.
train_batch_size = 1  # TODO: Set the training batch size.
val_batch_size = 1  # TODO: Set the validation batch size.

# TODO: Set pretrained model location
vgg_pretrained = '/home/codesteller/apsoft/models/pretrained/VGG-16_mod2FCN_ImageNet-Classification'
epochs = 6  # TODO: Set the number of epochs to train for.
