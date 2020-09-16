import tensorflow
import xml.etree.ElementTree as et
import dataset_util
import cv2
import os
import hashlib
class_id_file=open('class_ids.txt')
class_ids= {}
lines=class_id_file.readlines()
for line in lines:
    id=line.split(' ')[0]
    class_name=" ".join(line.split(' ')[1:])[:-1]
    class_ids[id]=class_name
print(class_ids)
os.mkdir("tfrecord")
os.mkdir("tfrecord/train")
os.mkdir("tfrecord/test")
os.mkdir("tfrecord/validation")
def read_xml_make_tfrecord():
    # train_writer = tensorflow.io.TFRecordWriter('train')
    # valid_writer = tensorflow.io.TFRecordWriter('valid')
    # test_writer = tensorflow.io.TFRecordWriter('test')
    num_data = 8
    for i in range(num_data):
        globals()['train_writer_{:05d}-of-{:05d}'.format(int(i), int(num_data))] = tensorflow.io.TFRecordWriter(
            'tfrecord/train/train.tfrecord-{:05d}-of-{:05d}'.format(
                int(i), int(num_data)))

    for i in range(int(num_data / 8)):
        globals()['test_writer_{:05d}-of-{:05d}'.format(int(i), int(num_data / 8))] = tensorflow.io.TFRecordWriter(
            'tfrecord/test/test.tfrecord-{:05d}-of-{:05d}'.format(
                int(i), int(num_data / 8)))
        globals()[
            'validation_writer_{:05d}-of-{:05d}'.format(int(i), int(num_data / 8))] = tensorflow.io.TFRecordWriter(
            'tfrecord/validation/validation.tfrecord-{:05d}-of-{:05d}'.format(
                int(i), int(num_data / 8)))
    length=len(os.listdir('data'))/2
    for number,xml_file in enumerate(os.listdir('data')):
        if xml_file[-4:]!='.xml' : continue
        filename=xml_file[:-4]

        tree=et.parse('data/'+filename+".xml")
        root=tree.getroot()

        height=int(root.find('imagesize').find('nrows').text)
        width=int(root.find('imagesize').find('ncols').text)

        # encoded_image_data=cv2.imread('imgs/'+filename)
        with tensorflow.io.gfile.GFile('data/'+filename+".jpg",'rb') as fid:
            encoded_image_data=fid.read()

        img_name=filename+".jpg"
        key = hashlib.sha256(encoded_image_data).hexdigest()
        image_format=filename.split('.')[-1]
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for object in root.findall('object'):

            object_name=object.find('name').text
            classes_text.append(object_name.encode('utf8'))
            classes.append(int(list(class_ids.keys())[list(class_ids.values()).index(object_name)]))

            points=[]
            for point in object.find('polygon').findall('pt'):
                x=int(point.find('x').text)
                y=int(point.find('y').text)
                points.append((x,y))

            xmins.append(min(points[i][0] for i in range(4)) /width)
            xmaxs.append(max(points[i][0] for i in range(4))/width)


            ymins.append(min(points[i][1] for i in range(4))/height)
            ymaxs.append(max(points[i][1] for i in range(4)) / height)
        print(img_name)
        print(xmins)
        print(xmaxs)
        print(ymins)
        print(ymaxs)
        print(classes_text)
        print(classes)
        example = tensorflow.train.Example(features=tensorflow.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(img_name.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(img_name.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            # 'image/object/difficult': dataset_util.int64_list_feature([0]),
            # 'image/object/truncated': dataset_util.int64_list_feature([0]),
            # 'image/object/view': dataset_util.bytes_list_feature(['Frontal'.encode('utf8')]),
            }))
        if number < length * 0.8:
            globals()[
                'train_writer_{:05d}-of-{:05d}'.format(int(number / (length * 0.8) * num_data), int(num_data))].write(
                example.SerializeToString())

        elif number < length * 0.9:
            globals()[
                'validation_writer_{:05d}-of-{:05d}'.format(int((number - length * 0.8) / (length * 0.1) * num_data / 8),
                                                            int(num_data / 8))].write(example.SerializeToString())
        elif number < length:

            globals()['test_writer_{:05d}-of-{:05d}'.format(int((number - length * 0.9) / (length * 0.1) * num_data / 8),
                                                            int(num_data / 8))].write(example.SerializeToString())


read_xml_make_tfrecord()