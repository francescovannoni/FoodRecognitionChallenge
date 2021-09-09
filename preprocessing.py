import os

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt


import collections

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

def import_data():
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
    return train_coco_inp, val_coco_inp

#remove images with mismatching dimensions
#There are rotated 27 images, unuseless, so change dataset or remove them (in book shows these images)

def check_badannotation(train_coco_inp):
    if not os.path.exists("data/train/annotations_correct.json"): #metti nel main
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

def check_non_empty_annotations(train_coco_inp):
    #checking if there are empty annotations
    count = 0
    for i in train_coco_inp["annotations"]:
        if not i["segmentation"][0]:
            count += 1
    print(count)

def check_if_odd_annotations(train_coco_inp):
    #checking if there are annotations with missing numbers
    count = 0
    for i in train_coco_inp["annotations"]:
        if len(i["segmentation"][0])%2 == 1:
            count += 1
    print(count)


def choosing_best_size(train_coco_inp):
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

# if you want try also mode, anyway probably best size may be 512 as it is the
# nearest power of two to the mean

def get_most_common(anns, N_MOST_COMMON): #get n most common categories and return dictionary
    #Loading into memory COCO dataset
    print(anns)

    ann_categories = []
    for a in anns:
        ann_categories.append(a["category_id"])

    counter = collections.Counter(ann_categories)
    most_common = counter.most_common(N_MOST_COMMON)
    most_common = dict(most_common)
    print(most_common)

    values = list(most_common.keys())

    #15 most frequent categories (dictionary 0....14 to cat_id)
    id_correspondence = {values[i]:i+1 for i in range(0, len(values))}
    return id_correspondence

def get_mask(coco_train, anns, most_common):
    img_to_masks = []
    folder = "data/train/images/"
    count = 0
    for filename in os.listdir(folder):
        img_id = int(filename.lstrip("0").rstrip(".jpg")) #getting image id
        img = cv2.imread(os.path.join(folder, filename))
        mask = np.zeros((img.shape[0], img.shape[1]))
        for ann in anns:
            if ann["image_id"]==img_id:
                if ann["category_id"] in most_common:
                    new_mask = most_common[ann["category_id"]]*coco_train.annToMask(ann)
                else:
                    new_mask = 16 * coco_train.annToMask(ann)
                for i in range(len(mask)):
                    for j in range(len(mask[0])):
                        if (new_mask[i][j]!=0 and mask[i][j]==0):
                            mask[i][j] = new_mask[i][j]
                        '''
                        elif (new_mask[i][j]!=0 and mask[i][j]!=0):
                            non_zero_new = np.count_nonzero(new_mask, keepdims=False)
                            non_zero_old = np.count_nonzero(mask == mask[i, j])
                            if non_zero_new < non_zero_old:
                                mask[i, j] = new_mask[i, j]
                                '''
        #overlap lasciato che modifica solo se non c'è già (prima). provare a vedere cosa cambia se si considera la più piccola
        img_to_masks.append((img, mask))

        mask = mask/16
        found= False
        for i in mask:
            for j in i:
                if j > 1:
                    found = True

        if found:
            count +=1

        plt.imshow(mask, cmap="gray")
        plt.show()

    print(count)

