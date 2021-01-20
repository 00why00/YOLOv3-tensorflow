import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

"""
将原始的 PASCAL 数据集转换为 TFRecord 以进行目标检测
参考 https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

使用方法:
    python tools/voc2012.py --data_dir ./data/VOCdevkit/VOC2012 --split train --output_file ./data/voc2012_train.tfrecord
    python tools/voc2012.py --data_dir ./data/VOCdevkit/VOC2012 --split val --output_file ./data/voc2012_val.tfrecord
"""

# 命令行参数
flags.DEFINE_string('data_dir', './data/VOCdevkit/VOC2012/', 'path to raw PASCAL VOC dataset')
flags.DEFINE_enum('split', 'train', ['train', 'val'], 'specify train or val spit')
flags.DEFINE_string('output_file', './data/voc2012_train.tfrecord', 'output dataset')
flags.DEFINE_string('classes', './data/voc2012.names', 'classes file')


def build_example(annotation, class_map):
    """
    参考 https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py#L59
    将 XML 转化为 tf.Example 类型
    :param annotation: 单个图片的 annotation
    :param class_map: 从 类别名 到 index 的映射
    :return: example: 已经转化的 tf.Example
    """
    img_path = os.path.join(FLAGS.data_dir, 'JPEGImages', annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def parse_xml(xml):
    """
    参考 https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py#L71
    递归地将 xml 转化为 dict 类型
    我们假设 object 标签是唯一可以在 xml tree 的同一级别中多次出现的标签
    :param xml: 使用 lxml etree 解析 xml 得到的 xml tree
    :return: 包含 xml 内容的 dict
    """
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# 参考 https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py#L147
def main(_argv):
    # 读取 类别 并生成 map
    class_map = {name: idx for idx, name in enumerate(open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    # 获得 train.txt / val.txt 文件里的全部图片名称列表
    image_list = open(os.path.join(FLAGS.data_dir, 'ImageSets', 'Main', '%s.txt' % FLAGS.split)).read().splitlines()
    logging.info("Image list loaded: %d", len(image_list))
    # 解析并转化
    for name in tqdm.tqdm(image_list):
        annotation_xml = os.path.join(FLAGS.data_dir, 'Annotations', name + '.xml')
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
