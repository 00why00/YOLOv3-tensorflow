from absl import app, flags, logging
from absl.flags import FLAGS
import os
import glob
import tensorflow as tf
from yolov3.models import (
    YoloV3, YoloV3Tiny
)
from yolov3.dataset import transform_images

"""
目标检测
检测数据集并将结果保存为 txt 文件
"""

flags.DEFINE_string('classes', './data/voc.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_7.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('path', './data/VOCdevkit/VOC2012/JPEGImages', 'path to image dir')
flags.DEFINE_integer('num_classes', 20, 'number of classes in the model')


def main(_argv):
    # 打开内存增长
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    # 创建网络模型
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)
    # 加载权重
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')
    # 加载类别列表
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    # 读取数据集
    root_path = os.getcwd()
    os.chdir(FLAGS.path)
    image_list = glob.glob('*.jpg')
    for image in image_list:
        # 处理图片
        img_raw = tf.image.decode_image(open(image, 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)
        # 对图片进行目标检测
        boxes, scores, classes, nums = yolo(img)
        # 保存结果
        save_file = os.path.join(root_path, 'data', 'detection_result', image.replace('.jpg', '.txt'))
        with open(save_file, 'a') as f:
            for i in range(nums[0]):
                obj_name = class_names[int(classes[0][i])]
                score = float('%.6f' % scores[0][i])
                left = int(boxes[0][i][0] * FLAGS.size)
                top = int(boxes[0][i][1] * FLAGS.size)
                right = int(boxes[0][i][2] * FLAGS.size)
                bottom = int(boxes[0][i][3] * FLAGS.size)
                f.write("%s %s %s %s %s %s\n" % (obj_name, score, left, top, right, bottom))

    logging.info('output saved complete')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
