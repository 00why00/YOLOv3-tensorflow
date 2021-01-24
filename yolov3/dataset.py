import tensorflow as tf
from absl.flags import FLAGS


# 将模型转换为易于部署且高性能的 TensorFlow 图模型
@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    """
    对 label 的 shape 进行转换
    原始shape (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    目标shape (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    :param y_true: 待转化 label
    :param grid_size: 13 / 26 / 52
    :param anchor_idxs: 每一个特征图对应的3个anchor的index
    :return:
    """
    num = tf.shape(y_true)[0]

    y_true_out = tf.zeros((num, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(num):
        for j in tf.range(tf.shape(y_true)[1]):
            # 忽略由 padding 产生的值为 0 的 box
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    # 更新指定位置
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    """
    调整 label
    :param y_train: 数据集 label
    :param anchors: anchor 数组
    :param anchor_masks: mask 数组
    :param size: 图片大小
    :return: 输出元组
    """
    y_outs = []
    grid_size = size // 32

    # 计算最好的 anchor 的 index
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    """
    调整图片
    :param x_train: 训练图片
    :param size: 调整到图片大小
    :return: 调整后的图片
    """
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


# 参考 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# 注释掉项目中不需要的
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table, size):
    """
    解析 tf record
    :param tfrecord: tf record
    :param class_table: class 的 dict
    :param size: 图片大小
    :return: x_train: 输入
    :return: y_train: 输出
    """
    # 解析 tf record 得到 dict
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)
    # 使 label 的大小和网络输出的大小(由 yolo max boxes 决定)相同，所以需要用 0 填充
    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size=416):
    """
    读取 tf record
    :param file_pattern: 数据集路径
    :param class_file: 类别文件路径
    :param size: 图片大小
    :return: dataset
    """
    # lookup 即 tensorflow 的 dict
    # 将 class 转化为 dict
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter="\n"), -1)
    # 读取 tf record 得到输入输出并建立映射
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


def load_fake_dataset():
    """
    当没有指定训练集时，加载此照片作为训练集
    :return: dataset
    """
    # 加载图片
    x_train = tf.image.decode_jpeg(open('./data/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)
    # 加载label
    labels = [
                 [0.18494931, 0.03049111, 0.9435849, 0.96302897, 0],
                 [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
                 [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
             ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
