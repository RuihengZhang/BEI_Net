import os
import shutil

def extract_matching_txt(images_dir, labels_dir, output_dir):
    # 创建输出目录的'test'子目录
    test_output_dir = os.path.join(output_dir, '')
    os.makedirs(test_output_dir, exist_ok=True)

    # 获取images目录下test子目录的所有图片文件名
    # test_image_files = [f for f in os.listdir(os.path.join(labels_dir, 'GT')) if os.path.isfile(os.path.join(labels_dir, 'GT', f))]
    test_image_files = [f for f in os.listdir(os.path.join(labels_dir)) if os.path.isfile(os.path.join(labels_dir, f))]
    # 提取匹配的txt文件，并移动到labels目录的test子目录下
    for image_file in test_image_files:
        txt_file = image_file.replace('.png', '.png')
        if os.path.exists(os.path.join(images_dir, txt_file)):
            shutil.copy(os.path.join(images_dir, txt_file), os.path.join(test_output_dir, txt_file))

# 示例用法
images_dir = '/home/kz/codes/代码/pred_results/myself'  # 图片目录
labels_dir = '/home/kz/codes/代码/test-index/CHANLLENGES/SO/GT'  # 标签目录
output_dir = '/home/kz/codes/代码/test-index/pred_results/myself/SO/GT'  # 输出目录

extract_matching_txt(images_dir, labels_dir, output_dir)
