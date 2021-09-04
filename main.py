# Downloading data (and exploring)

import os
import wget
import json
import cv2
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists("data"):
    os.mkdir("data")
    os.mkdir("data/train")
    os.mkdir("data/val")

with open('data/train/annotations.json') as json_file:
    train_coco_inp= json.load(json_file)
print("Keys of train annotation file: ",train_coco_inp.keys())
print("No of Categories(Classes) present: ",len(train_coco_inp['categories']))

with open('data/val/annotations.json') as json_file:
    val_coco_inp= json.load(json_file)

for i in val_coco_inp["annotations"]:
    if i["image_id"] == 81931:
        print(i["category_id"])

#get segmentation points
seg = val_coco_inp["annotations"][0]["segmentation"]
seg_chicken = val_coco_inp["annotations"][1]["segmentation"]

''' check which are the categories
for i in val_coco_inp["categories"]:
    if i["id"] == 1022:
        print(i["name"])
    if i["id"] == 1788:
        print(i["name"])
'''

img = cv2.imread('data/val/images/081931.jpg', cv2.COLOR_BGR2RGB)

seg_chicken = seg_chicken[0]
seg = seg[0]

print(seg_chicken)
x1 = (559, 185)
x2 = (429, 164)
x3 = (372, 255)
x4 = (443, 342)
x5 = (479,361)
x6 = (447,382)
x7 =(424,422)
x8 = (405,452)

img = cv2.circle(img, x1, 2, (255, 0, 0), 2)
img = cv2.circle(img, x2, 2, (255, 0, 0), 2)
img = cv2.circle(img, x3, 2, (255, 0, 0), 2)
img = cv2.circle(img, x4, 2, (255, 0, 0), 2)
img = cv2.circle(img, x5, 2, (255, 0, 0), 2)
img = cv2.circle(img, x6, 2, (255, 0, 0), 2)
img = cv2.circle(img, x7, 2, (255, 0, 0), 2)
img = cv2.circle(img, x8, 2, (255, 0, 0), 2)

y1 = (553, 496)
y1 = (362, 522)
y1 = (553, 496)
y1 = (553, 496)
y1 = (553, 496)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
useless = []
for i in train_coco_inp['images']:
  im = cv2.imread(f"data/train/images/{i['file_name']}")
  if((im.shape[0]!=i['height']) or (im.shape[1]!=i['width'])):
    #print(i, im.shape)
    useless.append(i)
  #else:
    #print(".",end="")

print(len(useless))

#There are rotated 27 images, unuseless, so change dataset or remove them (in book shows these images)

'''
