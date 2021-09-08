# Downloading data (and exploring)
import matplotlib.pyplot as plt

import preprocessing
import os
from pycocotools.coco import COCO

import numpy as np
import cv2

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
N_MOST_COMMON = 15

train_coco_inp, val_coco_inp = preprocessing.import_data()
if not os.path.exists("data/train/annotations_correct.json"):
    preprocessing.check_badannotation(train_coco_inp)

preprocessing.check_if_odd_annotations(train_coco_inp)
preprocessing.check_non_empty_annotations(train_coco_inp)
preprocessing.choosing_best_size(train_coco_inp)

#import coco
coco_train = COCO("data/train/annotations_correct.json")
cat_ids = coco_train.getCatIds()
categories = coco_train.loadCats(cat_ids)
names = [c["name_readable"] for c in categories]
print("Number of different categories: ", len(names))
ann_ids = coco_train.getAnnIds()
anns = coco_train.loadAnns(ann_ids)

#getting most common categories
most_common = preprocessing.get_most_common(anns, N_MOST_COMMON)
print(most_common)


#mask = np.zeros((128, 128))
preprocessing.get_mask(coco_train, anns)
mask = coco_train.annToMask(anns[55])

plt.imshow(mask, cmap="gray")
plt.show()