import os
import string
import xml.etree.ElementTree as et
import cv2  # ==4.
import numpy as np
import rstr
from PIL import Image
import shutil
import glob
import random


def image1OnImage2(img1, img2):
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))
    _, thresh = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV)
    img2 = cv2.bitwise_and(img2, img2, mask=thresh)
    return img1 + img2


def getRotatedImage(image):
    angle = np.random.randint(0, 360, size=1, dtype=np.int16)[0]
    print(angle)
    rotated = image.rotate(angle, resample=2, expand=True)
    rotated = np.array(rotated)
    rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    cv2.waitKey()
    h, w = rotated.shape[:2]
    size = int(max(h, w) * 1.2)
    # black background
    background = np.zeros((size, size, 3), dtype=np.uint8)
    randx = np.random.randint(0, size - w, size=1, dtype=np.int64)[0]
    randy = np.random.randint(0, size - h, size=1, dtype=np.int64)[0]

    background[randy:randy + h, randx:randx + w, :] = rotated
    image = background.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # x trước y sau
    box = np.concatenate([box, box], axis=0)

    if angle < 90:
        startPoint = 1
        points = box[startPoint:4 + startPoint]
    elif angle < 180:
        startPoint = 0
        points = box[startPoint:4 + startPoint]
    elif angle < 270:
        startPoint = 3
        points = box[startPoint:4 + startPoint]
    elif angle < 360:
        startPoint = 2
        points = box[startPoint:4 + startPoint]

    return image, points  # passport with black background and 4 points


def getPointsfromName(points):
    points = [int(i) for i in points.split(" ")]
    return np.reshape(points, (4, 2))


def passportOnReal(passport, points):
    h, w = passport.shape[:2]
    real_passport_path = random.choice(glob.glob("real_passport/*"))
    real_passport = cv2.imread(real_passport_path)
    real_points = getPointsfromName(real_passport_path[:-4].split("/")[-1])
    print(real_points, points)
    M = cv2.getPerspectiveTransform(np.array(real_points, dtype=np.float32), np.array(points, dtype=np.float32))
    real_passport = cv2.warpPerspective(real_passport, M, (w, h))
    return image1OnImage2(passport, real_passport)


def createBBox(x, y, h, w):  # tọa độ góc và size của passport
    ratio = 0.2
    xmin = max(x - int(w * ratio), 1)
    xmax = x + int(w * ratio)
    ymin = max(y - int(h * ratio), 1)
    ymax = y + int(h * ratio)
    return xmin, xmax, ymin, ymax


def createData():
    folder_name = "data/"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

    num_data = 5000
    while len(os.listdir(folder_name)) < num_data * 2:
        image_names = os.listdir('passport')
        image_name = image_names[np.random.randint(0, len(image_names), size=1, dtype=np.int64)[0]]
        if not image_name[-3:] == "jpg": continue
        image = Image.open("passport/" + image_name)
        w, h = image.size
        passport, points = getRotatedImage(image)
        passport = passportOnReal(passport, points)
        background = cv2.imread(random.choice(glob.glob("background/*")))
        image = image1OnImage2(passport, background)

        h_final, w_final = image.shape[:2]
        new_img_name = rstr.rstr(string.ascii_letters, 10)
        new_xml_file = et.parse('header.xml')
        new_root = new_xml_file.getroot()  # save root
        new_root.find("filename").text = new_img_name + ".jpg"
        new_root.find("imagesize").find("nrows").text = str(h_final)
        new_root.find("imagesize").find("ncols").text = str(w_final)
        for i, point in enumerate(points):
            object_names = ['top_left', 'top_right', 'bot_right', 'bot_left']
            x = point[0]
            y = point[1]
            add_element = et.parse('object.xml')
            add_object = add_element.getroot()
            add_object.find("name").text = object_names[i]
            xmin, xmax, ymin, ymax = createBBox(x, y, h, w)
            for i, pt in enumerate(add_object.find('polygon').findall('pt')):
                if i == 0 or i == 2:
                    pt.find('x').text = str(xmin)
                    pt.find('y').text = str(ymin)
                if i == 1 or i == 3:
                    pt.find('x').text = str(min(xmax, w_final - 1))
                    pt.find('y').text = str(min(ymax, h_final - 1))
            new_root.append(add_object)

        image = cv2.medianBlur(image, 5)
        cv2.imwrite(folder_name + new_img_name + '.jpg', image)
        new_xml_file.write(folder_name + new_img_name + '.xml')


createData()
