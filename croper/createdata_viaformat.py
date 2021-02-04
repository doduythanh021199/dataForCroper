from PIL import Image
import cv2
import numpy as np
import random
import glob
import shutil
import os
import rstr
import string
import json


def rotate_image(img):
    """
    :param img: np.ndarray
    :return:
    """
    angle = random.randint(-20, 20)
    if angle < 0: angle = 360 + angle
    img = Image.fromarray(img)
    img = img.rotate(angle=angle, expand=True)
    return np.asarray(img)


def image_on_background(img):
    h_img, w_img = img.shape[:2]
    ratio = random.randint(130, 150) / 100
    h = int(h_img * ratio)
    w = int(w_img * ratio)

    result = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    x_draw = random.randint(0, w - w_img)
    y_draw = random.randint(0, h - h_img)
    result[y_draw:y_draw + h_img, x_draw:x_draw + w_img, :] = img

    mask[cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) != 0] = 255
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    background = cv2.imread(random.choice(glob.glob("background/*.jpg")))
    background = cv2.resize(background, (w, h))
    background[mask != 0] = 0
    result = result + background

    return result, hull


def get_dict(passport, hull):
    """
    {"24631331976_defa3bb61f_k.jpg668058":
      {"fileref":"","size":668058,"filename":"24631331976_defa3bb61f_k.jpg","base64_img_data":"","file_attributes":{},
        "regions":{
                    "0":{"shape_attributes": {"name":"polygon",
                      "all_points_x":[916,913,905,889,868,836,809,792,789,784,777,769,767,777,786,791,769,739,714,678,645,615,595,583,580,584,595,614,645,676,716,769,815,849,875,900,916,916],
                      "all_points_y":[515,583,616,656,696,737,753,767,777,785,785,778,768,766,760,755,755,743,728,702,670,629,588,539,500,458,425,394,360,342,329,331,347,371,398,442,504,515]},
                      "region_attributes":{}}}},
    ......
    }
    """
    result = {}
    img_name = rstr.rstr(string.ascii_lowercase, 10) + ".jpg"
    print(img_name)
    h, w = passport.shape[:2]
    size = h * w
    all_points_x = list([int(x[0][0]) for x in hull])
    all_points_y = list([int(x[0][1]) for x in hull])
    dict = {"fileref": "", "size": size, "filename": img_name, "base64_img_data": "", "file_attributes": {},
            "regions": {
                "0": {
                    "shape_attributes": {"name": "polygon", "all_points_x": all_points_x, "all_points_y": all_points_y},
                    "region_attributes": {}}}}
    result[img_name + str(size)] = dict
    return img_name, result


def create_data_cocoformat(count):
    """
    count image for train and count/8 for val
    :param count:
    :return:
    """
    if os.path.exists("train"):
        shutil.rmtree("train")
    os.mkdir("train")
    if os.path.exists("val"):
        shutil.rmtree("val")
    os.mkdir("val")
    # for train
    dict_train = {}
    for _ in range(count):
        passport = cv2.imread(random.choice(glob.glob("real_passport/*.jpg")))
        # resize smaller
        h, w = passport.shape[:2]
        ratio = random.randint(200, 400) / 100
        passport = cv2.resize(passport, (int(w / ratio), int(h / ratio)))
        # rotate image
        rotated = rotate_image(passport)
        passport, hull = image_on_background(rotated)
        img_name, img_dict = get_dict(passport, hull)
        cv2.imwrite("train/" + img_name, passport)
        dict_train.update(img_dict)
    with open('train/via_region_data.json', 'w') as fp:
        json.dump(dict_train, fp)

    dict_val = {}
    for _ in range(int(count / 8)):
        passport = cv2.imread(random.choice(glob.glob("real_passport/*.jpg")))
        # resize smaller
        h, w = passport.shape[:2]
        ratio = random.randint(200, 400) / 100
        passport = cv2.resize(passport, (int(w / ratio), int(h / ratio)))
        # rotate image
        rotated = rotate_image(passport)
        passport, hull = image_on_background(rotated)
        img_name, img_dict = get_dict(passport, hull)
        cv2.imwrite("val/" + img_name, passport)
        dict_val.update(img_dict)
    with open('val/via_region_data.json', 'w') as fp:
        json.dump(dict_val, fp)


create_data_cocoformat(80)
