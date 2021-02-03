import sys
import os
import glob
import lxml.etree as ET

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 后退两级目录得到项目根目录
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
# 设置解析文件保存目录
save_path = os.path.join(parent_path, 'data', 'ground_truth')
# 得到 xml 文件目录
ground_truth_path = os.path.join(parent_path, 'data', 'VOCdevkit', 'VOC2012', 'Annotations')
os.chdir(ground_truth_path)
# 解析 xml 文件
xml_list = glob.glob('*.xml')
if len(xml_list) == 0:
    print("没有 xml 文件")
    sys.exit()
for xml in xml_list:
    save_file = os.path.join(save_path, xml.replace('.xml', '.txt'))
    with open(save_file, 'a') as f:
        root = ET.parse(xml).getroot()
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
print("转化完成！")

