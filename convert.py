from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3.models import YoloV3, YoloV3Tiny
from yolov3.utils import load_darknet_weights
import tensorflow as tf

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        # 启动内存增长（如果为启用内存增长，则运行时初始化不会分配设备上的所有内存）
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 创建 网络模型
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.summary()
    logging.info('model created')
    # 加载预训练权重
    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info('weights loaded')
    # 检查
    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')
    # 保存权重
    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
