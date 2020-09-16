import os
import string
import xml.etree.ElementTree as et
import cv2  #==4.
import numpy as np
import rstr
from PIL import Image
size=1000
def getRotatedImage(image):
    angle=np.random.randint(0,360,size=1,dtype=np.int16)[0]
    print(angle)
    rotated=image.rotate(angle,expand=True)
    rotated=np.array(rotated)
    rotated=cv2.cvtColor(rotated,cv2.COLOR_BGR2RGB)
    cv2.waitKey()
    h,w=rotated.shape[:2]
    #black background

    background=np.zeros((size,size,3),dtype=np.uint8)
    randx=np.random.randint(0,size-w,size=1,dtype=np.int64)[0]
    randy=np.random.randint(0,size-h,size=1,dtype=np.int64)[0]

    background[randy:randy+h,randx:randx+w,:]=rotated
    image=background.copy()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rect=cv2.minAreaRect(contours[0])
    box=cv2.boxPoints(rect)
    box=np.int0(box)# x trước y sau
    box=np.concatenate([box,box],axis=0)

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

    return image,points    #passport with black background and 4 points

def imageWithRealGround(image):
    background_names=os.listdir("background")
    randomGround_name=background_names[np.random.randint(0,len(background_names),size=1,dtype=np.int64)[0]]
    background=cv2.imread("background/"+randomGround_name)
    background=cv2.resize(background,(size,size))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)

    background=cv2.bitwise_and(background,background,mask=thresh)
    image=background+image
    image=cv2.GaussianBlur(image,(3,3),0)
    return image

def createBBox(x,y,h,w):# tọa độ góc và size của passport
    xmin=max(x-int(w*0.1),0)
    xmax=x+int(w*0.1)
    ymin=max(y-int(h*0.1),0)
    ymax=y+int(h*0.1)
    return xmin,xmax,ymin,ymax

def createData():
    folder_name="data/"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    num_data=20
    while len(os.listdir(folder_name)) < num_data:
        image_names=os.listdir('passport')
        image_name=image_names[np.random.randint(0,len(image_names),size=1,dtype=np.int64)[0]]
        if not image_name[-3:]=="jpg": continue
        image=Image.open("passport/"+image_name)
        w,h=image.size
        if w>=size/1.5 or h>=size/1.5: continue
        image,points=getRotatedImage(image)
        image=imageWithRealGround(image)

        new_img_name = rstr.rstr(string.ascii_letters, 10)
        new_xml_file = et.parse('header.xml')
        new_root = new_xml_file.getroot()  # save root
        new_root.find("filename").text = new_img_name + ".jpg"
        new_root.find("imagesize").find("nrows").text = str(size)
        new_root.find("imagesize").find("ncols").text = str(size)
        for i,point in enumerate(points):
            object_names=['top_left','top_right','bot_right','bot_left']
            x=point[0]
            y=point[1]
            add_element = et.parse('object.xml')
            add_object = add_element.getroot()
            add_object.find("name").text = object_names[i]
            xmin,xmax,ymin,ymax=createBBox(x,y,h,w)
            for i, pt in enumerate(add_object.find('polygon').findall('pt')):
                if i == 0 or i == 2:
                    pt.find('x').text = str(xmin)
                    pt.find('y').text = str(ymin)
                if i == 1 or i == 3:
                    pt.find('x').text = str(min(xmax,size))
                    pt.find('y').text = str(min(ymax,size))
            new_root.append(add_object)
        cv2.imwrite(folder_name+new_img_name+'.jpg',image)
        new_xml_file.write(folder_name+new_img_name+'.xml')


createData()