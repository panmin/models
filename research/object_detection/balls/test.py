from fileinput import filename
import os
import sys
import random
import xml.etree.ElementTree as ET
import tensorflow as tf

def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
    values: A scalar or list of values.
    Returns:
    a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
 
def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
 
def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _process_image(directory, name):
    filename = os.path.join(directory, name)
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    tree = ET.parse(filename)
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)]
    # Find annotations.
    # 获取每个object的信息
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(0)
        labels_text.append(label.encode('ascii'))
 
        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)
 
        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    print(image_data, shape, bboxes, labels, labels_text, difficult, truncated)
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated

def _convert_to_example(image_data, labels, labels_text, bboxes, shape, difficult, truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned
 
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example

def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, 
                                  labels,
                                  labels_text,
                                  bboxes, 
                                  shape, 
                                  difficult, 
                                  truncated)
    tfrecord_writer.write(example.SerializeToString())


def run(dir, output_dir, shuffling=False):
    # 如果output_dir不存在则创建
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    # train.txt
    split_file_path = os.path.join(dir,'train.txt')
    with open(split_file_path) as f:
        filenames = f.readlines()
    # shuffling == Ture时，打乱顺序
    if shuffling:
        random.seed(4242)
        random.shuffle(filenames)
    # Process dataset files.
    i = 0
    fidx = 0
    dataset_dir = os.path.join(dir)
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = '%s/%03d.tfrecord' % (output_dir, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < 4330:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()
                filename = filenames[i].strip()
                _add_to_tfrecord(dataset_dir, filename, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\n>> Finished converting the Pascal VOC dataset!')


if __name__ == '__main__':
    dir = '/Users/panmin/tensorflow/models2/research/object_detection/balls/'
    run(dir,os.path.join(dir,"output"))
    # filename = dir+"train.txt"
    # with open(filename, 'r') as f1:
    #     list1 = f1.readlines()
    # for i in range(0, len(list1)):
    #     list1[i] = list1[i].rstrip('\n')
    #     image_anno = list1[i].split(" ")
    #     image = image_anno[0]
    #     anno = image_anno[1]
    #     _deal_anno(os.path.join(dir,anno))






# def _process_image():
    # # Read the image file.
    # filename = os.path.join(directory, DIRECTORY_IMAGES, name + '.jpg')
    # image_data = tf.gfile.FastGFile(filename, 'r').read()
    # # Read the XML annotation file.
    # filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    # tree = ET.parse(filename)
    # root = tree.getroot()
    # # Image shape.
    # size = root.find('size')
    # shape = [int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)]
    # # Find annotations.
    # # 获取每个object的信息
    # bboxes = []
    # labels = []
    # labels_text = []
    # difficult = []
    # truncated = []
    # for obj in root.findall('object'):
    #     label = obj.find('name').text
    #     labels.append(int(VOC_LABELS[label][0]))
    #     labels_text.append(label.encode('ascii'))
 
    #     if obj.find('difficult'):
    #         difficult.append(int(obj.find('difficult').text))
    #     else:
    #         difficult.append(0)
    #     if obj.find('truncated'):
    #         truncated.append(int(obj.find('truncated').text))
    #     else:
    #         truncated.append(0)
 
    #     bbox = obj.find('bndbox')
    #     bboxes.append((float(bbox.find('ymin').text) / shape[0],
    #                    float(bbox.find('xmin').text) / shape[1],
    #                    float(bbox.find('ymax').text) / shape[0],
    #                    float(bbox.find('xmax').text) / shape[1]
    #                    ))
    # return image_data, shape, bboxes, labels, labels_text, difficult, truncated
