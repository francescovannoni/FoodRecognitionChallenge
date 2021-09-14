# Downloading data (and exploring)
import random

import matplotlib.pyplot as plt

import preprocessing
import unet
from tensorflow.keras import callbacks
import os
from pycocotools.coco import COCO
import pickle

import numpy as np
import cv2

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
N_MOST_COMMON = 14
EPOCHS = 5
BATCH_SIZE = 256

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

if os.path.exists("X_train.pickle") and os.path.exists("y_train.pickle"):
    with open('X_train.pickle', 'rb') as handle:
        X_train = pickle.load(handle)
    with open('y_train.pickle', 'rb') as handle:
        y_train = pickle.load(handle)
else:
    X_train, y_train = preprocessing.train_generator(coco_train, anns, most_common)
print(X_train.shape)

model = unet.unet_model()
model_checkpoint = callbacks.ModelCheckpoint('model.h5', monitor="val_accuracy", save_best_only=True)

callbacks = [
    callbacks.EarlyStopping(patience=4, monitor='accuracy'),  # monitor era val_loss
    callbacks.TensorBoard(log_dir='logs')
]

random.shuffle(X_train)
random.shuffle(y_train)

results = model.fit(X_train[:512], y_train[:512], batch_size=4, epochs=EPOCHS, callbacks=callbacks)

model.save('/model/')

'''
preds_train = model.predict(X_train[0])
plt.imshow(preds_train, cmap='gray')
plt.show()
plt.imshow(y_train[0], cmap='gray')
'''

# plt.imshow(mask, cmap="gray")
# plt.show()
