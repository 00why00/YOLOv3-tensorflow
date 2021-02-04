import os
import sys
import shutil
import glob
import json
import operator
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from absl import app, flags, logging
from absl.flags import FLAGS

"""
计算 mAP
"""

flags.DEFINE_boolean('no_animation', True, 'no animation is shown')
flags.DEFINE_boolean('no_plot', True, 'no plot is shown')
flags.DEFINE_boolean('quiet', False, 'minimalistic console output')
# e.g. python mAP.py -ignore "person book"
flags.DEFINE_spaceseplist('ignore', None, 'ignore a list of classes')
# e.g. python mAP.py -set_class_iou "person 0.7 book 0.6"
flags.DEFINE_spaceseplist('set_class_iou', None, 'set IoU for a specific class')

# 见 http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#4.4
MINOVERLAP = 0.5

'''
    0,0 -------------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
     |      |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


def calculate_log_average_miss_rate(precision, recall):
    """
    在区间 [1e-2, 1]以对数空间均匀分为 9 份，计算平均的 MR
    :param precision: 精确率 TP / (TP + FP) = TP / n
    :param recall: 召回率 TP / (TP + FN) = TP / P
    :return: lamr: log-average miss rate
    :return: mr: miss rate MR = FN / (TP + FN) = FN / P
    :return: fppi: false positive per image FPPI = FP / (TP + FP) = FP / n
    """
    # 如果没有此类的预测
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = 1 - precision
    mr = 1 - recall

    # 在起止位置插入值防止越界
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # 将[1e-2, 1e0]在对数空间均匀分为 9 份
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # 因为 ref 的最小值为 0.01，fppi_tmp 的最小值为 -1.0
        # 所以一定可以找到至少一个索引
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def check_float_between_0_and_1(value):
    """
    检查数字是不是一个在0和1之间的浮点数
    """
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def voc_ap(precision, recall):
    """
    参考：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#3.4
    根据 precision 和 recall 数组计算 AP：
        1、计算随着 precision 递减的 Precision-recall 曲线
        2、AP 就是 Precision-recall 曲线下面的面积
    代码参考 VOC development kit code 中 VOCap.m
    :param precision: 精确率 TP / (TP + FP) = TP / n
    :param recall: 召回率 TP / (TP + FN) = TP / P
    :return: ap: average-precision 平均精度
    :return: precision: 横轴
    :return: recall: 纵轴
    """
    # 在 precision 和 recall 数组前后插值
    precision.insert(0, 0.0)
    precision.append(0.0)
    recall.insert(0, 0.0)
    recall.append(1.0)
    # 让 precision 单调递减（从后往前）
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    # 记录 recall 改变的位置
    change_list = []
    for i in range(1, len(recall)):
        if recall[i] != recall[i - 1]:
            change_list.append(i)
    # 使用数值积分计算 Precision-recall 曲线下面的面积
    ap = 0.0
    for i in change_list:
        ap += ((precision[i] - recall[i - 1]) * precision[i])
    return ap, precision, recall


def file_lines_to_list(path):
    """
    将文件按行保存成列表
    """
    with open(path) as f:
        content = f.readlines()
    # 去除每一行末尾的 空格 或 \n
    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text, pos, color, line_width):
    """
    在图片上写字
    """
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    line_type = 1
    bottom_left_corner_of_text = pos
    cv2.putText(img, text, bottom_left_corner_of_text, font, font_scale, color, line_type)
    text_width, _ = cv2.getTextSize(text, font, font_scale, line_type)[0]
    return img, (line_width + text_width)


def adjust_axes(renderer, text, fig, axes):
    """
    调整 plot 坐标轴
    """
    # 计算文字宽度用于重新缩放
    box = text.get_window_extent(renderer=renderer)
    text_width_inches = box.width / fig.dpi
    # 计算缩放比例
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    # 设置坐标轴最大值
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])


def draw_plot_func(dictionary, num_classes, window_title, plot_title, x_label, output_path, if_show, plot_color, tp_bar):
    """
    使用 Matplotlib 绘图
    """
    # 降序排列字典的值到元组列表中
    sort_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # 解压元组列表为两个列表
    sorted_keys, sorted_values = zip(*sort_dic_by_value)
    # 有 TP 数组时
    if tp_bar != "":
        """
        绿色：TP
        红色：FP
        粉色：FN
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - tp_bar[key])
            tp_sorted.append(tp_bar[key])
        # 绘制水平直方图
        plt.barh(range(num_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(num_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        plt.legend(loc='lower right')
        # 标数值
        fig = plt.gcf()
        axes = plt.gca()
        renderer = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            text = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):
                adjust_axes(renderer, text, fig, axes)
    else:
        # 绘制水平直方图
        plt.barh(range(num_classes), sorted_values, color=plot_color)
        # 标数值
        fig = plt.gcf()
        axes = plt.gca()
        renderer = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            text = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):
                adjust_axes(renderer, text, fig, axes)
    # 设置窗口标题
    fig.canvas.set_window_title(window_title)
    # 在 y 轴上写类名
    tick_font_size = 12
    plt.yticks(range(num_classes), sorted_keys, fontsize=tick_font_size)
    # 相应的缩放高度
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = num_classes * (tick_font_size * 1.4)  # 1.4 为间距
    height_in = height_pt / dpi
    top_margin = 0.15  # 百分比
    bottom_margin = 0.15  # 百分比
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)
    # 设置图标题
    plt.title(plot_title, fontsize=14)
    # 设置坐标轴名称
    plt.xlabel(x_label, fontsize='large')
    # 适应窗口大小
    fig.tight_layout()
    # 保存图表
    fig.savefig(output_path)
    # 展示
    if if_show:
        plt.show()
    plt.close()


def main(_argv):
    # 检查是否有要忽略的类别
    if FLAGS.ignore is None:
        FLAGS.ignore = []

    # 设置文件路径
    ground_truth_path = os.path.join(os.getcwd(), '../data', 'ground_truth')
    detection_results_path = os.path.join(os.getcwd(), '../data', 'detection_result')
    image_path = os.path.join(os.getcwd(), '../data', 'VOCdevkit', 'VOC2012', 'JPEGImages')

    # 没有图片时设置 no_animation 为 True
    if os.path.exists(image_path):
        for root, dirs, files in os.walk(image_path):
            if not files:
                FLAGS.no_animation = True
    else:
        FLAGS.no_animation = True

    # 创建 temp 和 output 目录
    temp_file_path = '.temp'
    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)
    output_file_path = 'output'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    else:
        shutil.rmtree(output_file_path)
        os.makedirs(output_file_path)
    if not FLAGS.no_plot:
        os.makedirs(os.path.join(output_file_path, 'classes'))
    if not FLAGS.no_animation:
        os.makedirs(os.path.join(output_file_path, 'images', 'detections_one_by_one'))

    """
    获取并解析 ground truth 文件
    """
    # 得到 ground truth 文件的列表
    ground_truth_file_list = glob.glob(ground_truth_path + '/*.txt')
    if len(ground_truth_file_list) == 0:
        logging.error("没有找到ground truth文件!")
        sys.exit(0)

    ground_truth_file_list.sort()
    ground_truth_counter_per_class = {}
    image_counter_per_class = {}

    ground_truth_file = []
    for txt_file in ground_truth_file_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # 检查是否有对应的 detection result 文件
        result_path = os.path.join(detection_results_path, (file_id + '.txt'))
        if not os.path.exists(result_path):
            error_msg = "没有找到: {}\n".format(result_path)
            logging.error(error_msg)
            sys.exit(0)
        lines_list = file_lines_to_list(txt_file)
        # 创建 ground truth 的字典
        bounding_boxes = []
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
                    is_difficult = False
            except ValueError:
                error_msg = txt_file + "格式错误"
                logging.error(error_msg)
                sys.exit(0)
            # 检查是否有忽略的类
            if class_name in FLAGS.ignore:
                continue
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name": class_name,
                                   "bbox": bbox,
                                   "used": False,
                                   "difficult": is_difficult})
            # 对于难识别的物体，不计算
            if not is_difficult:
                # 对每个类的标记进行计数
                if class_name in ground_truth_counter_per_class:
                    ground_truth_counter_per_class[class_name] += 1
                else:
                    ground_truth_counter_per_class[class_name] = 1
                # 对每个类的图片进行计数
                if class_name not in already_seen_classes:
                    if class_name in image_counter_per_class:
                        image_counter_per_class[class_name] += 1
                    else:
                        image_counter_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # 将 bounding box 保存为 json 文件
        temp_file = temp_file_path + "/" + file_id + "_ground_truth.json"
        ground_truth_file.append(temp_file)
        with open(temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    ground_truth_classes = list(ground_truth_counter_per_class.keys())
    ground_truth_classes = sorted(ground_truth_classes)
    num_classes = len(ground_truth_classes)

    # 检查是否设置了某个类别的 IoU
    iou_list = []
    if FLAGS.set_class_iou is not None:
        num_args = len(FLAGS.set_class_iou)
        if num_args % 2 != 0:
            logging.error("输入参数个数不为2的倍数！")
            sys.exit(0)
        special_iou_classes = FLAGS.set_class_iou[::2]
        iou_list = FLAGS.set_class_iou[1::2]
        for tmp_class in special_iou_classes:
            if tmp_class not in ground_truth_classes:
                logging.error("未知的类：" + tmp_class)
                sys.exit(0)
        for tmp_iou in iou_list:
            if not check_float_between_0_and_1(tmp_iou):
                logging.error("错误的IoU值：" + tmp_iou)
                sys.exit(0)

    """
    获取并解析 detection result 文件
    """
    # 得到 detection result 文件的列表
    detection_results_file_list = glob.glob(detection_results_path + '/*.txt')
    detection_results_file_list.sort()

    for class_index, class_name in enumerate(ground_truth_classes):
        bounding_boxes = []
        for txt_file in detection_results_file_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # 检查是否有对应的 ground truth 文件
            truth_path = os.path.join(ground_truth_path, (file_id + '.txt'))
            if class_index == 0 and not os.path.exists(truth_path):
                error_msg = "没有找到: {}\n".format(truth_path)
                logging.error(error_msg)
                sys.exit(0)
            lines_list = file_lines_to_list(txt_file)
            for line in lines_list:
                try:
                    predict_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = txt_file + "格式错误"
                    logging.error(error_msg)
                    sys.exit(0)
                if predict_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence,
                                           "bbox": bbox,
                                           "file_id": file_id})
        # 根据置信度降序排列 detection result
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        # 保存为 json 文件
        with open(temp_file_path + '/' + class_name + "_detection_result.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
    计算每个类的 AP
    """
    sum_ap = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # 保存输出
    with open(output_file_path + "/output.txt", 'w') as output_file:
        output_file.write("# AP and precision/recall per class\n")
        true_positive_count = {}
        for class_index, class_name in enumerate(ground_truth_classes):
            true_positive_count[class_name] = 0
            # 加载对应类的 detection result
            detection_results_file = temp_file_path + '/' + class_name + "_detection_result.json"
            detection_results_data = json.load(open(detection_results_file))

            num_data = len(detection_results_data)
            tp = [1e-6] * num_data
            fp = [1e-6] * num_data
            for index, detection in enumerate(detection_results_data):
                file_id = detection["file_id"]
                if not FLAGS.no_animation:
                    # 找到对应图片
                    ground_truth_image = glob.glob1(image_path, file_id + ".*")
                    if len(ground_truth_image) == 0:
                        logging.error("没有找到图片：" + file_id)
                        sys.exit(0)
                    elif len(ground_truth_image) > 1:
                        logging.error("找到多张图片：" + file_id)
                        sys.exit(0)
                    else:
                        # 加载图片
                        img = cv2.imread(image_path + '/' + ground_truth_image[0])
                        # 加载带有预测框的图片
                        img_cumulative_path = output_file_path + "/images/" + ground_truth_image[0]
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        # 给图片添加底边
                        bottom_border = 60
                        black = [0, 0, 0]
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=black)
                # 加载对应的 ground truth 文件
                ground_truth_file = temp_file_path + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(ground_truth_file))
                iou_max = -1
                ground_truth_match = -1
                # 加载预测框
                bounding_box_dr = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    # 查看类别名是否匹配
                    if obj["class_name"] == class_name:
                        # 加载标记框
                        bounding_box_gt = [float(x) for x in obj["bbox"].split()]
                        bounding_box_intersection = [max(bounding_box_dr[0], bounding_box_gt[0]),
                                                     max(bounding_box_dr[1], bounding_box_gt[1]),
                                                     max(bounding_box_dr[2], bounding_box_gt[2]),
                                                     max(bounding_box_dr[3], bounding_box_gt[3])]
                        intersection_width = bounding_box_intersection[2] - bounding_box_intersection[0] + 1
                        intersection_height = bounding_box_intersection[3] - bounding_box_intersection[1] + 1
                        if intersection_width > 0 and intersection_height > 0:
                            union = (bounding_box_dr[2] - bounding_box_dr[0] + 1) * \
                                    (bounding_box_dr[3] - bounding_box_dr[1] + 1) + \
                                    (bounding_box_gt[2] - bounding_box_gt[0] + 1) * \
                                    (bounding_box_gt[3] - bounding_box_gt[1] + 1) - \
                                    intersection_width * intersection_height
                            iou = intersection_width * intersection_height / union
                            if iou > iou_max:
                                iou_max = iou
                                ground_truth_match = obj
                # 认为识别结果为 TP
                if not FLAGS.no_animation:
                    status = "没有找到匹配项!"
                min_overlap = MINOVERLAP
                if FLAGS.set_class_iou is not None:
                    if class_name in FLAGS.set_class_iou:
                        iou_index = FLAGS.set_class_iou.index(class_name)
                        min_overlap = float(iou_list[iou_index])
                if iou_max >= min_overlap:
                    if "difficult" not in ground_truth_match:
                        if not bool(ground_truth_match["used"]):
                            # TP
                            tp[index] = 1
                            ground_truth_match["used"] = True
                            true_positive_count[class_name] += 1
                            # 更新 json 文件
                            with open(ground_truth_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                            if not FLAGS.no_animation:
                                status = "匹配成功!"
                        else:
                            # FP 多次识别
                            fp[index] = 1
                            if iou_max > 0:
                                status = "重复匹配!"
                else:
                    # FP
                    fp[index] = 1
                    if iou_max > 0:
                        status = "overlap 不足"

                # 显示动画
                if not FLAGS.no_animation:
                    height, width = img.shape[:2]
                    # 颜色 BGR
                    white = (255, 255, 255)
                    light_blue = (255, 200, 100)
                    green = (0, 255, 0)
                    light_red = (30, 30, 255)
                    # 第一条线
                    margin = 10
                    v_pos = int(height - margin - (bottom_border / 2.0))
                    text = "Image: " + ground_truth_image[0] + " "
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text = "Class [" + str(class_index) + "/" + str(num_classes) + "]: " + class_name + " "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                         line_width)
                    if iou_max != -1:
                        color = light_red
                        if status == "overlap 不足":
                            text = "IoU: {0:.2f}% ".format(iou_max * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                        else:
                            text = "IoU: {0:.2f}% ".format(iou_max * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                            color = green
                        img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    # 第二条线
                    v_pos += int(bottom_border / 2.0)
                    rank_pos = str(index + 1)
                    text = "Detection #rank: " + rank_pos + \
                           " confidence: {0:.2f}% ".format(float(detection["confidence"]) * 100)
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    color = light_red
                    if status == "匹配成功!":
                        color = green
                    text = "Result: " + status + " "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # 如果预测框和标记框有交集
                    if iou_max > 0:
                        bounding_box_gt = [int(round(float(x))) for x in ground_truth_match["bbox"].split()]
                        cv2.rectangle(img, (bounding_box_gt[0], bounding_box_gt[1]),
                                      (bounding_box_gt[2], bounding_box_gt[3]), light_blue, 2)
                        cv2.rectangle(img_cumulative, (bounding_box_gt[0], bounding_box_gt[1]),
                                      (bounding_box_gt[2], bounding_box_gt[3]), light_blue, 2)
                        cv2.putText(img_cumulative, class_name, (bounding_box_gt[0], bounding_box_gt[1] - 5),
                                    font, 0.6, light_blue, 1, cv2.LINE_AA)
                    bounding_box_dr = [int(i) for i in bounding_box_dr]
                    cv2.rectangle(img, (bounding_box_dr[0], bounding_box_dr[1]),
                                  (bounding_box_dr[2], bounding_box_dr[3]), color, 2)
                    cv2.rectangle(img_cumulative, (bounding_box_dr[0], bounding_box_dr[1]),
                                  (bounding_box_dr[2], bounding_box_dr[3]), color, 2)
                    cv2.putText(img_cumulative, class_name, (bounding_box_dr[0], bounding_box_dr[1] - 5),
                                font, 0.6, color, 1, cv2.LINE_AA)
                    # 展示图片
                    cv2.imshow("Animation", img)
                    cv2.waitKey(20)
                    # 保存图片
                    output_image_path = (output_file_path + "/images/detections_one_by_one/" +
                                         class_name + "_detection" + str(index) + ".jpg")
                    cv2.imwrite(output_image_path, img)
                    cv2.imwrite(img_cumulative_path, img_cumulative)

            # 计算 precision / recall
            cumsum = 0
            for index, val in enumerate(fp):
                fp[index] += cumsum
                cumsum += val
            cumsum = 0
            for index, val in enumerate(tp):
                tp[index] += cumsum
                cumsum += val
            # TODO: 使用 numpy 代替 list
            recall = tp[:]
            for index, val in enumerate(tp):
                # noinspection PyTypeChecker
                recall[index] = float(tp[index]) / ground_truth_counter_per_class[class_name]
            precision = tp[:]
            for index, val in enumerate(tp):
                # noinspection PyTypeChecker
                precision[index] = float(tp[index]) / (fp[index] + tp[index])

            ap, m_recall, m_precision = voc_ap(precision[:], recall[:])
            sum_ap += ap
            text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "

            # 写入输出文件
            rounded_precision = ['%.2f' % elem for elem in precision]
            rounded_recall = ['%.2f' % elem for elem in recall]
            output_file.write(text + "\n Precision: " + str(rounded_precision) + "\n Recall :" + str(rounded_recall) + "\n\n")
            if not FLAGS.quiet:
                print(text)
            ap_dictionary[class_name] = ap

            _num_images = image_counter_per_class[class_name]
            lamr, mr, fppi = calculate_log_average_miss_rate(np.array(precision), np.array(recall))
            lamr_dictionary[class_name] = lamr

            # 画图表
            if not FLAGS.no_plot:
                plt.plot(recall, precision, '-o')
                # 在 list 倒数第二位置添加一点 (m_recall[-2], 0.0)，因为最后的一段不影响 AP 的值
                area_under_curve_x = m_recall[:-1] + [m_recall[-2] + m_recall[-1]]
                area_under_curve_y = m_precision[:-1] + [0.0] + [m_precision[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # 设置窗口标题
                fig = plt.gcf()
                fig.canvas.set_window_title('AP ' + class_name)
                # 设置图表标题
                plt.title('class:' + text)
                # 设置坐标轴名称
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # 设置坐标轴
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                # 保存图表
                fig.savefig(output_file_path + "/classes/" + class_name + ".png")
                plt.cla()

        if not FLAGS.no_animation:
            cv2.destroyAllWindows()

        output_file.write("\n# mAP of all classes\n")
        m_ap = sum_ap / num_classes
        text = "mAP = {0:.2f}%".format(m_ap * 100)
        output_file.write(text + "\n")
        print(text)

    # FP
    if not FLAGS.no_animation:
        pink = (203, 192, 255)
        for tmp_file in ground_truth_file:
            ground_truth_data = json.load(open(tmp_file))
            start = temp_file_path + '/'
            img_id = tmp_file[tmp_file.find(start) + len(start): tmp_file.rfind('_ground_truth.json')]
            img_cumulative_path = output_file_path + "/images/" + img_id + ".jpg"
            img = cv2.imread(img_cumulative_path)
            if img is None:
                img_path = image_path + '/' + img_id + ".jpg"
                img = cv2.imread(img_path)
            # 画 FP
            for obj in ground_truth_data:
                if not obj['used']:
                    bounding_box_gt = [int(round(float(x))) for x in obj["bbox"].split()]
                    cv2.rectangle(img, (bounding_box_gt[0], bounding_box_gt[1]),
                                  (bounding_box_gt[2], bounding_box_gt[3]), pink, 2)
            cv2.imwrite(img_cumulative_path, img)

    # 计算 detection result 总数
    detection_result_counter_per_class = {}
    for txt_file in detection_results_file_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in FLAGS.ignore:
                continue
            if class_name in detection_result_counter_per_class:
                detection_result_counter_per_class[class_name] += 1
            else:
                detection_result_counter_per_class[class_name] = 1
    detection_result_classes = list(detection_result_counter_per_class.keys())

    # 做 ground truth 中每个类的个数的图表
    if not FLAGS.no_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_file_list)) + " files and " + str(num_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = output_file_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(ground_truth_counter_per_class, num_classes, window_title, plot_title, x_label, output_path,
                       to_show, plot_color, '')

    # 保存 ground truth 中每个类的个数到 output.txt
    with open(output_file_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(ground_truth_counter_per_class):
            output_file.write(class_name + ": " + str(ground_truth_counter_per_class[class_name]) + "\n")

    # 完成对 TP 计数
    for class_name in detection_result_classes:
        # 如果在 detection result 但是没有在 ground truth 里，说明这个类里没有 TP
        if class_name not in ground_truth_classes:
            true_positive_count[class_name] = 0

    # 做 detection result 中每个类的个数的图表
    if not FLAGS.no_plot:
        window_title = "detection-results-info"
        plot_title = "detection-results\n"
        plot_title += "(" + str(len(detection_results_file_list)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(detection_result_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        x_label = "Number of objects per class"
        output_path = output_file_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_positive_bar = true_positive_count
        draw_plot_func(detection_result_counter_per_class, len(detection_result_counter_per_class), window_title,
                       plot_title, x_label, output_path, to_show, plot_color, true_positive_bar)

    # 保存 detection result 中每个类的个数到 output.txt
    with open(output_file_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(detection_result_classes):
            output_file.write(class_name + ": " + str(detection_result_counter_per_class[class_name]) +
                              " (tp:" + str(true_positive_count[class_name]) + ", fp:" +
                              str(detection_result_counter_per_class[class_name] - true_positive_count[class_name]) +
                              ")\n")

    # log-average miss rate
    if not FLAGS.no_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = output_file_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(lamr_dictionary, num_classes, window_title, plot_title, x_label, output_path, to_show,
                       plot_color, "")

    # mAP
    if not FLAGS.no_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(m_ap * 100)
        x_label = "Average Precision"
        output_path = output_file_path + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(ap_dictionary, num_classes, window_title, plot_title, x_label, output_path, to_show,
                       plot_color, "")

    # 删除 temp 文件夹
    shutil.rmtree(temp_file_path)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
