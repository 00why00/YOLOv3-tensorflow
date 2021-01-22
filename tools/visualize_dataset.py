from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np

from yolov3.dataset import load_tfrecord_dataset
from yolov3.utils import draw_outputs

"""
可视化数据集中的一张图片
参考 https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

使用方法:
     ̶p̶y̶t̶h̶o̶n̶ ̶t̶o̶o̶l̶s̶/̶v̶i̶s̶u̶a̶l̶i̶z̶e̶_̶d̶a̶t̶a̶s̶e̶t̶.̶p̶y̶ ̶-̶-̶c̶l̶a̶s̶s̶e̶s̶=̶.̶/̶d̶a̶t̶a̶/̶v̶o̶c̶2̶0̶1̶2̶.̶n̶a̶m̶e̶s̶
    使用 CMD 运行会找不到父目录中的模块，可以直接在 pycharm 中运行
"""

flags.DEFINE_string('classes', '../data/coco.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('dataset', '../data/voc2012_train.tfrecord', 'path to dataset')
flags.DEFINE_string('output', '../output.jpg', 'path to output image')
flags.DEFINE_integer('yolo_max_boxes', 100, 'maximum number of boxes per image')


def main(_argv):
    # 读取 类别 并生成 list
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    # 读取 TF record格式 的数据集并 随机排序
    dataset = load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes, FLAGS.size)
    dataset = dataset.shuffle(512)
    # 读取第一张图片并可视化并保存
    for image, labels in dataset.take(1):
        boxes = []
        scores = []
        classes = []
        for x1, y1, x2, y2, label in labels:
            if x1 == 0 and x2 == 0:
                continue

            boxes.append((x1, y1, x2, y2))
            scores.append(1)
            classes.append(label)
        nums = [len(boxes)]
        boxes = [boxes]
        scores = [scores]
        classes = [classes]

        logging.info('labels:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(FLAGS.output, img)
        logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    app.run(main)
