import os
import string
import xml.etree.ElementTree as et
import cv2  #==4.
import numpy as np
import rstr
from PIL import Image
size=4000
def imageWithRealGround(image,realpassport):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)
    realpassport=cv2.bitwise_and(realpassport,realpassport,mask=thresh)
    image=realpassport+image
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image

def getRealPassport(image,realpassport_name):
    realpassport = cv2.imread("real_passport/" + realpassport_name)
    h,w=realpassport.shape[:2]
    points=realpassport_name[:-4].split(" ")
    points=[int(i) for i in points]
    [tl,tr,br,bl]=np.reshape(np.array(points),newshape=(4,2))
    widthA = np.sqrt(((br[0] - bl[0]) * (br[0] - bl[0])) + ((br[1] - bl[1]) * (br[1] - bl[1])))
    widthB = np.sqrt(((tr[0] - tl[0]) * (tr[0] - tl[0])) + ((tr[1] - tl[1]) * (tr[1] - tl[1])))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) * (tr[0] - br[0])) + ((tr[1] - br[1]) * (tr[1] - br[1])))
    heightB = np.sqrt(((tl[0] - bl[0]) * (tl[0] - bl[0])) + ((tl[1] - bl[1]) * (tl[1] - bl[1])))
    maxHeight = max(int(heightA), int(heightB))

    image=cv2.resize(image,(maxWidth,maxHeight))

    points = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    dst=[tl,tr,br,bl]
    points = np.float32(points)
    dst = np.float32(dst)
    M = cv2.getPerspectiveTransform(points, dst)
    image = cv2.warpPerspective(image, M, (w,h))

    image=Image.fromarray(image)
    realpassport=Image.fromarray(realpassport)

    return image,realpassport

def getRotatedImage(image,realpassport):
    angle = np.random.randint(0, 360, size=1, dtype=np.int16)[0]
    image=image.rotate(angle,expand=True)
    image=np.array(image)

    realpassport = realpassport.rotate(angle, expand=True)
    realpassport = np.array(realpassport)


    # background=np.zeros((size,size,3),dtype=np.uint8)
    # randx=np.random.randint(0,size-w,size=1,dtype=np.int64)[0]
    # randy=np.random.randint(0,size-h,size=1,dtype=np.int64)[0]
    #
    # background[randy:randy+h,randx:randx+w,:]=image
    # image=background.copy()
    # background[randy:randy + h, randx:randx + w, :] = realpassport
    # realpassport=background.copy()

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # rect=cv2.minAreaRect(contours[0])
    # box=cv2.boxPoints(rect)
    # box=np.int0(box)# x trước y sau
    # box=np.concatenate([box,box],axis=0)
    cnt=contours[0]
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    box=approx.ravel()
    # print(box)
    # img_copy=image.copy()
    # cv2.drawContours(img_copy, [approx], -1, (0, 255, 0), 3)
    # cv2.imshow("img",cv2.resize(img_copy,(500,500)))
    # cv2.waitKey()
    box=np.reshape(box,newshape=(4,2))
    box = np.int0(box)  # x trước y sau

    box=np.concatenate([box,box],axis=0)
    box=box[2:6]
    box=[box[0],box[3],box[2],box[1]]
    box = np.concatenate([box, box], axis=0)

    if angle<90:
        startPoint=1
        points=box[startPoint:4+startPoint]

    elif angle<180:
        startPoint=0
        points=box[startPoint:4+startPoint]

    elif angle<270:
        startPoint=3
        points=box[startPoint:4+startPoint]

    elif angle < 360:
        startPoint = 2
        points = box[startPoint:4+startPoint]

    # img_copy = image.copy()
    # for i,point in enumerate(points):
    #     cv2.circle(img_copy, (point[0], point[1]), 5, (0,255 , 0), 5)
    #     cv2.putText(img_copy,str(i+1),(point[0], point[1]),cv2.FONT_HERSHEY_COMPLEX,10,(255,0,0),5)
    # cv2.imshow("img", cv2.resize(img_copy, (int(img_copy.shape[1]/4),int(img_copy.shape[0]/4))))
    # cv2.waitKey()
    return imageWithRealGround(image,realpassport),points  #passport with black background and 4 points

def createBBox(x,y,h,w):# tọa độ góc và size của passport
    ratio=0.05
    xmin=max(x-int(w*ratio),0)
    xmax=x+int(w*ratio)
    ymin=max(y-int(h*ratio),0)
    ymax=y+int(h*ratio)
    return xmin,xmax,ymin,ymax

def createData():
    folder_name = "data/"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    num_data = 1000
    while len(os.listdir(folder_name)) < num_data*2:
        image_names = os.listdir('passport')
        image_name = image_names[np.random.randint(0, len(image_names), size=1, dtype=np.int64)[0]]
        realpassport_names = os.listdir("real_passport")
        realpassport_name = realpassport_names[np.random.randint(0, len(realpassport_names), size=1, dtype=np.int64)[0]]
        background_names = os.listdir("background")
        randomGround_name = background_names[np.random.randint(0, len(background_names), size=1, dtype=np.int64)[0]]
        background = cv2.imread("background/" + randomGround_name)

        if not image_name[-3:] == "jpg": continue
        image = cv2.imread("passport/" + image_name)
        # h,w = image.shape[:2]#size of passport
        # if w >= size / 1.5 or h >= size / 1.5: continue

        image,realpassport=getRealPassport(image,realpassport_name)
        w,h=image.size#size of passport

        image,points=getRotatedImage(image,realpassport)


        hf,wf=image.shape[:2]#size of final image

        background = cv2.resize(background, (wf,hf))
        image=imageWithRealGround(image,background)


        new_img_name = rstr.rstr(string.ascii_letters, 10)
        new_xml_file = et.parse('header.xml')
        new_root = new_xml_file.getroot()  # save root
        new_root.find("filename").text = new_img_name + ".jpg"
        new_root.find("imagesize").find("nrows").text = str(hf)
        new_root.find("imagesize").find("ncols").text = str(wf)

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
                    pt.find('x').text = str(min(xmax, size))
                    pt.find('y').text = str(min(ymax, size))
            new_root.append(add_object)

        cv2.imwrite(folder_name + new_img_name + '.jpg', image)
        new_xml_file.write(folder_name + new_img_name + '.xml')


createData()