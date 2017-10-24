#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple
import numpy as np


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,        0 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,        0 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        0 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        0 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        0 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        0 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        0 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,        0 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        0 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        0 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,        0 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

LABELS = labels

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Pretty-print function for `labels`
#--------------------------------------------------------------------------------

# Print function for `labels`
def print_labels(labels):
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>5} | {:>8} | {:>12} | {:>15}".format( 'name', 'id', 'trainId', 'category', 'catId', 'hasInsts', 'ignoreInEval', 'color' ))
    print("    " + ('-' * 111))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>5} | {:>8} | {:>12} | {:>15}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval, str(label.color)) )
    print("")

#--------------------------------------------------------------------------------
# Some useful conversion dictionaries and arrays as global variables
#--------------------------------------------------------------------------------

ids_to_colors_dict = {label.id: label.color for label in labels}
colors_to_ids_dict = {label.color: label.id for label in labels}
trainIds_to_colors_dict = {label.trainId: label.color for label in labels}
colors_to_trainIds_dict = {label.color: label.trainId for label in labels}
ids_to_trainIds_dict = {label.id: label.trainId for label in labels}
trainIds_to_ids_dict = {label.trainId: label.id for label in labels}
ids_to_categoryIds_dict = {label.id: label.categoryId for label in labels}
categoryIds_to_ids_dict = {label.categoryId: label.id for label in labels}

ids_to_colors_array = np.zeros(shape=(35, 3), dtype=np.uint8)
for class_id, color in ids_to_colors_dict.items():
    ids_to_colors_array[class_id] = color

ids_to_trainIds_array = np.zeros(shape=35, dtype=np.uint8)
for old_id, new_id in ids_to_trainIds_dict.items():
    ids_to_trainIds_array[old_id] = new_id

trainIds_to_ids_array = np.zeros(shape=20, dtype=np.uint8)
for old_id, new_id in trainIds_to_ids_dict.items():
    trainIds_to_ids_array[old_id] = new_id
trainIds_to_ids_array[0] = 0

ids_to_categoryIds_array = np.zeros(shape=35, dtype=np.uint8)
for old_id, new_id in ids_to_categoryIds_dict.items():
    ids_to_categoryIds_array[old_id] = new_id

categoryIds_to_ids_array = np.zeros(shape=8, dtype=np.uint8)
for old_id, new_id in categoryIds_to_ids_dict.items():
    categoryIds_to_ids_array[old_id] = new_id

IDS_TO_COLORS_DICT = ids_to_colors_dict
COLORS_TO_IDS_DICT = colors_to_ids_dict
TRAINIDS_TO_COLORS_DICT = trainIds_to_colors_dict
COLORS_TO_TRAINIDS_DICT = colors_to_trainIds_dict
IDS_TO_TRAINIDS_DICT = ids_to_trainIds_dict
TRAINIDS_TO_IDS_DICT = trainIds_to_ids_dict
IDS_TO_CATEGORYIDS_DICT = ids_to_categoryIds_dict
CATEGORYIDS_TO_IDS_DICT = categoryIds_to_ids_dict

IDS_TO_COLORS_ARRAY = ids_to_colors_array
IDS_TO_TRAINIDS_ARRAY = ids_to_trainIds_array
TRAINIDS_TO_IDS_ARRAY = trainIds_to_ids_array
IDS_TO_CATEGORYIDS_ARRAY = ids_to_categoryIds_array
CATEGORYIDS_TO_IDS_ARRAY = categoryIds_to_ids_array

IDS_TO_RGBA_DICT = {key: (*value, 127) for key, value in IDS_TO_COLORS_DICT.items()}
TRAINIDS_TO_RGBA_DICT = {key: (*value, 127) for key, value in TRAINIDS_TO_COLORS_DICT.items()}

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print_labels(labels)

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
