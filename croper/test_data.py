import xml.etree.ElementTree as et
import os
import cv2
folder_name="data/"
for number,xml_file in enumerate(os.listdir(folder_name)):
        if xml_file[-4:]!='.xml' : continue
        filename=xml_file[:-4]

        tree=et.parse(folder_name+filename+".xml")
        root=tree.getroot()

        height=int(root.find('imagesize').find('nrows').text)
        width=int(root.find('imagesize').find('ncols').text)
        print(filename)
        image=cv2.imread(folder_name+filename+'.jpg')
        cv2.imshow('truowc',image)

        for object in root.findall('object'):

            object_name=object.find('name').text

            points=[]
            for point in object.find('polygon').findall('pt'):
                x=int(point.find('x').text)
                y=int(point.find('y').text)
                points.append((x,y))
            points=list(set(points))
            points=sorted(points)
            cv2.rectangle(image,points[0],points[1],(2550,0,0),2)
            cv2.putText(image,object_name,points[0],cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('concu',image)
        cv2.waitKey()