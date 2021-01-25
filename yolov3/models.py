from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .utils import broadcast_iou

flags.DEFINE_integer('yolo_max_boxes', 100, 'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')

# 参考 https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
# 参考 https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    """
    Darknet-53 卷积层
    :param x: 输入
    :param filters: 过滤器个数
    :param size: 卷积核的大小
    :param strides: 步长
    :param batch_norm: 是否使用 BatchNormalization
    :return: x: 输出
    """
    if strides == 1:
        padding = 'same'
    else:
        # 左上方填充
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    """
    Darknet-53 残差层
    :param x: 输入
    :param filters: 过滤器个数
    :return: x: 输出
    """
    # shortcut
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    # 连接
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    """
    Darknet-53 一个卷积 + 多个残差层
    :param x: 输入
    :param filters: 过滤器个数
    :param blocks: 残差层个数
    :return: x: 输出
    """
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    """
    Darknet-53
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet53.cfg
    :param name: 网络名
    :return: 网络
    """
    x = inputs = Input([None, None, 3])  # batch_size, img_size, img_size, 3
    x = DarknetConv(x, 32, 3)  # batch_size, img_size, img_size, 32
    x = DarknetBlock(x, 64, 1)  # batch_size, img_size, img_size, 64
    x = DarknetBlock(x, 128, 2)  # batch_size. img_size, img_size, 128
    x = x_36 = DarknetBlock(x, 256, 8)  # batch_size, img_size, img_size, 256 提取特征,用于将97和36层特征拼接在一起
    x = x_61 = DarknetBlock(x, 512, 8)  # batch_size, img_size, img_size, 512 提取特征，用于将85和61层特征拼接在一起
    x = DarknetBlock(x, 1024, 4)  # batch_size, img_size, img_size, 1024
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None):
    """
    Darknet-53 tiny 即 YOLO v3 tiny 去掉最后三层卷积
    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
    :param name: 网络名
    :return: 网络
    """
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # 跳跃连接
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, name=None):
    """
    YOLO v3 卷积层
    :param filters: 过滤器个数
    :param name: 层名
    :return: 网络
    """
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # 上采样
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloConvTiny(filters, name=None):
    """
    YOLO v3 tiny 卷积层
    :param filters: 过滤器个数
    :param name: 层名
    :return: 网络
    """
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # 上采样
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    """
    YOLO v3 输出层
    :param filters: 过滤器个数
    :param anchors: anchor数组
    :param classes: 类别数
    :param name: 网络名
    :return: 输出网络
    """
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])  # batch_size, 13/26/52, 13/26/52, 512/256/128
        x = DarknetConv(x, filters * 2, 3)  # batch_size, 13/26/52, 13/26/52, 1024/512/256
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)  # batch_size, 13/26/52, 13/26/52, 255
        x = Lambda(lambda _x: tf.reshape(_x, (-1, tf.shape(_x)[1], tf.shape(_x)[2], anchors, classes + 5)))(x)  # batch_size, grid, grid, 3, 85
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


def yolo_boxes(pred, anchors, classes):
    """
    解析预测结果
    :param pred: 预测值 (batch_size, grid, grid, anchors, (x, y, w, h, obj, classes))
    :param anchors: anchor数组
    :param classes: 类别数
    :return: bbox: bounding box 坐标 (x1 y1 x2 y2)
    :return objectness: 1 | 0 | -1 见论文
    :return class_probs: 类别置信度
    :return pred_box: bounding box 坐标 (x y w h)
    """
    # 得到特征图大小
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)  # batch_size, grid, grid, 3, 2
    objectness = tf.sigmoid(objectness)  # batch_size, grid, grid, 3, 1
    class_probs = tf.sigmoid(class_probs)  # batch_size, grid, grid, 3, 80
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # batch_size, grid, grid, 3, 4

    # grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs):
    """
    NMS 非极大值抑制
    :param outputs: 输出 (boxes, conf, type)
    :return: boxes: bounding box
    :return: scores: 置信度
    :return: classes: 类别
    :return: valid_detections: 有效 bbox 个数
    """
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    """
    YOLO v3
    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    :param size: 输入图片大小
    :param channels: 输入图片维度
    :param anchors: anchor 数组
    :param masks: mask 数组
    :param classes: 类别个数
    :param training: 是否训练
    :return: 网络
    """
    x = inputs = Input([size, size, channels], name='input')  # batch_size, 416, 416, 3

    # x: batch_size, 13, 13, 1024; x_61: batch_size, 26, 26, 512; x_36: batch_size, 52, 52, 256
    x_36, x_61, x = Darknet(name='yolo_darknet')(x)
    # 第一层特征图，用于预测大尺寸物体
    x = YoloConv(512, name='yolo_conv_0')(x)  # None, 13, 13, 512
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)  # batch_size, grid, grid, 3, 85
    # 第二层特征图，用于预测中尺寸物体
    x = YoloConv(256, name='yolo_conv_1')((x, x_61))  # None, 26, 26, 256
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)  # batch_size, grid, grid, 3, 85
    # 第三层特征图，用于预测小尺寸物体
    x = YoloConv(128, name='yolo_conv_2')((x, x_36))  # None, 52, 52, 128
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)  # batch_size, grid, grid, 3, 85

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda _x: yolo_boxes(_x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda _x: yolo_boxes(_x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda _x: yolo_boxes(_x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)
    outputs = Lambda(lambda _x: yolo_nms(_x), name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    """
    YOLO v3 tiny
    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
    :param size: 输入图片大小
    :param channels: 输入图片维度
    :param anchors: anchor 数组
    :param masks: mask 数组
    :param classes: 类别个数
    :param training: 是否训练
    :return: 网络
    """
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)
    # 第一层特征图，用于预测大尺寸物体
    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)
    # 第一层特征图，用于预测小尺寸物体
    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda _x: yolo_boxes(_x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda _x: yolo_boxes(_x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda _x: yolo_nms(_x), name='yolo_nms')((boxes_0[:3], boxes_1[:3]))

    return Model(inputs, outputs, name='yolov3_tiny')


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    """
    损失函数
    :param anchors: anchor 数组
    :param classes: 类别数
    :param ignore_thresh: iou 阈值
    :return: 损失
    """
    def yolo_loss(y_true, y_pred):
        """
        计算损失
        :param y_true: 真实值 (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, classes))
        :param y_pred: 预测值 (batch_size, grid, grid, anchors, (x, y, w, h, obj, classes))
        :return: 损失
        """
        # 1. 解析预测值
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. 解析真实值
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # 提高小 box 的权重
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. 转化真实值的 xy 为相对的，即转换到[0, 1]区间; wh 计算 bbox 的 log 大小
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. 计算全部的 anchor
        obj_mask = tf.squeeze(true_obj, -1)
        # 当 IoU 大于阈值时忽略假正类（false positive）
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. 计算全部的损失
        # 中心坐标误差
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        # 宽高坐标误差
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        # 置信度误差（交叉熵）
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # 分类误差（交叉熵）
        # 当分类一种物体时使用 binary_crossentropy
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        # 6. 求和 (batch, grid_x, grid_y, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
