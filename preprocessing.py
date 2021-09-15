import os

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
import random
import pickle

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


def import_data():
    if not os.path.exists("data"):
        os.mkdir("data")
        os.mkdir("data/train")
        os.mkdir("data/val")

    # load train annotations
    with open('data/train/annotations.json') as json_file:
        train_coco_inp = json.load(json_file)
    print("Keys of train annotation file: ", train_coco_inp.keys())
    print("No of Categories(Classes) present: ", len(train_coco_inp['categories']))

    # load val annotations
    with open('data/val/annotations.json') as json_file:
        val_coco_inp = json.load(json_file)
    print("Keys of val annotation file: ", val_coco_inp.keys())
    print("No of Categories(Classes) present: ", len(val_coco_inp['categories']))
    return train_coco_inp, val_coco_inp

# remove images with mismatching dimensions
# There are rotated 27 images, unuseless, so change dataset or remove them (in book shows these images)
def check_badannotation(coco_inp, train_val):
    useless = []
    for i in coco_inp['images']:
        im = cv2.imread("data/" + train_val + f"/images/{i['file_name']}")
        if (im.shape[0] != i['height']) or (im.shape[1] != i['width']):
            os.remove("data/" + train_val + f"/images/{i['file_name']}")
            useless.append(i)

    print("Number of images with mismatching dimensions: ", len(useless))
    bad_ids = [item["id"] for item in useless]
    for i, item in enumerate(coco_inp['images']):
        if item["id"] in bad_ids:
            del coco_inp["images"][i]

    for i, item in enumerate(coco_inp['annotations']):
        if item["id"] in bad_ids:
            del coco_inp["annotations"][i]

    with open("data/" + train_val + "/annotations_correct.json", "w") as f:
        f.write(json.dumps(coco_inp))

def check_non_empty_annotations(train_coco_inp):
    # checking if there are empty annotations
    count = 0
    for i in train_coco_inp["annotations"]:
        if not i["segmentation"][0]:
            count += 1
    print('Numbers of empty annotations: ', count)

def check_if_odd_annotations(train_coco_inp):
    # checking if there are annotations with missing numbers
    count = 0
    for i in train_coco_inp["annotations"]:
        if len(i["segmentation"][0]) % 2 == 1:
            count += 1
    print('Numbers of odd annotations: ', count)

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

def get_most_common(anns, N_MOST_COMMON):  # get n most common categories and return dictionary
    # Loading into memory COCO dataset

    ann_categories = []
    for a in anns:
        ann_categories.append(a["category_id"])

    counter = collections.Counter(ann_categories)
    most_common = counter.most_common(N_MOST_COMMON)
    most_common = dict(most_common)
    print(most_common)

    values = list(most_common.keys())

    # 14 most frequent categories (dictionary 1....14 to cat_id, 15 others, 0 background)
    id_correspondence = {values[i]: 15-i for i in range(0, len(values))}
    return id_correspondence

def get_mask(img_id, img_shape, coco_train, anns, most_common):
    mask = np.zeros((img_shape[0], img_shape[1]))
    for ann in anns:
        if ann["image_id"] == img_id:
            if ann["category_id"] in most_common:
                new_mask = most_common[ann["category_id"]] * coco_train.annToMask(ann)
            else:
                new_mask = coco_train.annToMask(ann) #no one of the most common
            mask = np.maximum(new_mask, mask)
        # provare a vedere cosa cambia se si considera la più piccola
        # alternative: togliere completamente quelle con ovelap oppure come alex
    mask = np.expand_dims(cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)), axis=-1)  # resizing all images
    mask = mask / 15
    return mask




def generator(coco_data, anns, most_common, path, train: bool):
    list_imgs = []
    for filename in tqdm(os.listdir(path)):
        list_imgs.append(filename)
    dataset_size = len(list_imgs)
    X = np.zeros((dataset_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    y = np.zeros((dataset_size, IMG_HEIGHT, IMG_WIDTH, 1))

    for i in range(dataset_size):
        img_id = int(list_imgs[i].lstrip("0").rstrip(".jpg"))  # getting image id
        img = cv2.imread(os.path.join(path, list_imgs[i]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        y[i] = get_mask(img_id, img.shape, coco_data, anns, most_common)
        X[i] = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if train:
        with open('X_train.pickle', 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('y_train.pickle', 'wb') as handle:
            pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('X_test.pickle', 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('y_val.pickle', 'wb') as handle:
            pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return X, y

'''
def generator(coco_data, anns, most_common, BATCH_SIZE):
    folder = "data/train/images/"

    #prima erano liste
    X_train = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    y_train = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1))
    i=0
    list_imgs = []
    for filename in tqdm(os.listdir(folder)):
        list_imgs.append(filename)
    random.shuffle(list_imgs)
    dataset_size = len(list_imgs)
    print("Number of images: ", dataset_size)

    while True:
        for j in range(i, i+BATCH_SIZE):
            img_id = int(list_imgs[j].lstrip("0").rstrip(".jpg"))  # getting image id
            img = cv2.imread(os.path.join(folder, list_imgs[j]))
            y_train[j-i] = (get_mask(img_id, img.shape, coco_data, anns, most_common))
            X_train[j-i] = (cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)))

        i += BATCH_SIZE
        if(i+BATCH_SIZE >= dataset_size):
            i=0
            random.shuffle(list_imgs)
        yield X_train, y_train
        '''
