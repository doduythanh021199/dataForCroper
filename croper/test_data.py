import cv2
import json
import numpy as np

data = json.load(open("train/via_region_data.json"))
print(data)
for key, value in data.items():
    img_name = value['filename']
    print(img_name)
    img = cv2.imread("train/" + img_name)
    print(img.shape)
    contours = []
    for region in value['regions'].items():
        region = region[1]
        all_points_x = region['shape_attributes']['all_points_x']
        all_points_y = region['shape_attributes']['all_points_y']
        print(type(all_points_y))
        contour = []
        for i in range(len(all_points_x)):
            contour.append([[all_points_x[i], all_points_y[i]]])
        contours.append(np.array(contour,dtype=int))
    cv2.drawContours(img, contours, -1, (255, 0, 0), 5)
    cv2.imshow("asd", img)
    cv2.waitKey()
