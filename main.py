# Downloading data (and exploring)
import random
import numpy as np
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
    preprocessing.check_badannotation(train_coco_inp, "train")
if not os.path.exists("data/val/annotations_correct.json"):
    preprocessing.check_badannotation(val_coco_inp, "val")  # badannotation generale da mettere


preprocessing.check_if_odd_annotations(train_coco_inp)
preprocessing.check_non_empty_annotations(train_coco_inp)
preprocessing.choosing_best_size(train_coco_inp)

# import coco
# TRAINING DATA
coco_train = COCO("data/train/annotations_correct.json")
train_cat_ids = coco_train.getCatIds()
categories = coco_train.loadCats(train_cat_ids)
names = [c["name_readable"] for c in categories]
print("Number of different categories: ", len(names))
train_ann_ids = coco_train.getAnnIds()
train_anns = coco_train.loadAnns(train_ann_ids)

# VALIDATION DATA
coco_val = COCO("data/val/annotations_correct.json")
val_cat_ids = coco_val.getCatIds()
val_ann_ids = coco_val.getAnnIds()
val_anns = coco_val.loadAnns(val_ann_ids)

# getting most common categories
most_common = preprocessing.get_most_common(train_anns, N_MOST_COMMON)

if os.path.exists("X_train.pickle") and os.path.exists("y_train.pickle"):
    with open('X_train.pickle', 'rb') as handle:
        X_train = pickle.load(handle)
    with open('y_train.pickle', 'rb') as handle:
        y_train = pickle.load(handle)
else:
    X_train, y_train = preprocessing.generator(coco_train, train_anns, most_common, path="data/train/images/",
                                               train=True)

if os.path.exists("X_val.pickle") and os.path.exists("y_val.pickle"):
    with open('X_val.pickle', 'rb') as handle:
        X_val = pickle.load(handle)
    with open('y_val.pickle', 'rb') as handle:
        y_val = pickle.load(handle)
else:
    X_val, y_val = preprocessing.generator(coco_val, val_anns, most_common, path="data/val/images/", train=False)

model = unet.unet_model()
model_checkpoint = callbacks.ModelCheckpoint('model.h5', monitor="val_accuracy", save_best_only=True)

callbacks = [
    callbacks.EarlyStopping(patience=4, monitor='accuracy'),  # monitor era val_loss
    callbacks.TensorBoard(log_dir='logs')
]

random.shuffle(X_train)
random.shuffle(y_train)

results = model.fit(X_train[:512], y_train[:512], batch_size=4, epochs=EPOCHS, callbacks=callbacks)

model.save('model/')

#try some elements
X_val = X_val[:10]
y_val = y_val[:10]
prediction = model.predict(X_val)
print(X_val.shape)

for i in range(len(X_val)):
    plt.imshow(X_val[i].astype(np.uint8))
    plt.show()
    plt.imshow(y_val[i])
    plt.show()
    plt.imshow(np.round(prediction[i]))
    plt.show()
