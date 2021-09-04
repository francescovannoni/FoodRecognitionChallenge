# Downloading data (and exploring)

import os
import numpy as np
import json
import cv2
from pycocotools.coco import COCO

if not os.path.exists("data"):
    os.mkdir("data")
    os.mkdir("data/train")
    os.mkdir("data/val")

#load train annotations
with open('data/train/annotations.json') as json_file:
    train_coco_inp= json.load(json_file)
print("Keys of train annotation file: ",train_coco_inp.keys())
print("No of Categories(Classes) present: ",len(train_coco_inp['categories']))

#load val annotations
with open('data/val/annotations.json') as json_file:
    val_coco_inp= json.load(json_file)
print("Keys of val annotation file: ", val_coco_inp.keys())
print("No of Categories(Classes) present: ",len(val_coco_inp['categories']))

#remove images with mismatching dimensions
#There are rotated 27 images, unuseless, so change dataset or remove them (in book shows these images)

if not os.path.exists("data/train/annotations_correct.json"):

    useless = []
    for i in train_coco_inp['images']:
      im = cv2.imread(f"data/train/images/{i['file_name']}")
      if (im.shape[0] != i['height']) or (im.shape[1] != i['width']):
        useless.append(i)

    print("Number of images with mismatching dimensions: ", len(useless))
    bad_ids = [item["id"] for item in useless]
    for i, item in enumerate(train_coco_inp['images']):
      if item["id"] in bad_ids:
        del train_coco_inp["images"][i]

    for i, item in enumerate(train_coco_inp['annotations']):
      if item["id"] in bad_ids:
        del train_coco_inp["annotations"][i]

    with open("data/train/annotations_correct.json", "w") as f:
      f.write(json.dumps(train_coco_inp))

#Loading into memory COCO dataset
cat_train = COCO("data/train/annotations_correct.json")
cat_ids = cat_train.getCatIds()
categories = cat_train.loadCats(cat_ids)

names = [c["name_readable"] for c in categories]

print("Number of different categories: ", len(names))

#checking if there are empty annotations
count = 0
for i in train_coco_inp["annotations"]:
    if not i["segmentation"][0]:
        count += 1
print(count)

#checking if there are annotations with missing numbers
count = 0
for i in train_coco_inp["annotations"]:
    if len(i["segmentation"][0])%2 == 1:
        count += 1
print(count)

img_width = []
img_height = []

for item in train_coco_inp["images"]:
    img_width.append(item["width"])
    img_height.append(item["height"])

max_width, min_width = max(img_width), min(img_width)
max_height, min_height = max(img_height), min(img_height)

mean_width = np.mean(img_width)
mean_height = np.mean(img_height)

print("max_width: ", max_width)
print("min_width: ", min_width)
print("mean_width: ", mean_width)
print("max_height: ", max_height)
print("min_height: ", min_height)
print("mean_height: ", mean_height)

#if you want try also mode, anyway probably best size may be 512 as it is the
#nearest power of two to the mean







