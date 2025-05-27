import os
import shutil
import cv2
import numpy as np
import math
import tifffile as tiff
import argparse

def generate_sliding_window_images(img_path, img_crop_path, window_size, desired_overlap):
    window_size = window_size
    desired_overlap = desired_overlap
    step_size = int(window_size * (1 - desired_overlap))
    
    img_name = os.path.basename(img_path).split(".")[0]
    img = cv2.imread(img_path)
    
    height, width, _ = img.shape
    if height < window_size or width < window_size:
        pad_bottom = max(0, window_size - height)
        pad_right = max(0, window_size - width)
        img = cv2.copyMakeBorder(
            img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    
    y_steps = list(range(0, img.shape[0] - window_size, step_size))
    x_steps = list(range(0, img.shape[1] - window_size, step_size))
    
    if img.shape[0] % window_size != 0 or img.shape[1] % window_size != 0:
        padded_height = math.ceil(img.shape[0] / window_size) * window_size
        padded_width = math.ceil(img.shape[1] / window_size) * window_size
        pad_bottom = padded_height - img.shape[0]
        pad_right = padded_width - img.shape[1]
        img = cv2.copyMakeBorder(
            img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    y_steps = list(range(0, img.shape[0] - window_size + 1, step_size))
    x_steps = list(range(0, img.shape[1] - window_size + 1, step_size))

    count_y = 0
    for y in y_steps:
        count_x = 0
        for x in x_steps:
            is_retained = True
            window_x1 = x
            window_y1 = y
            window_x2 = x + window_size
            window_y2 = y + window_size
            
            if is_retained:
                window = img[window_y1:window_y2, window_x1:window_x2]
                file_path = os.path.join(img_crop_path, f"{img_name}_{count_y}{count_x}.png")
                cv2.imwrite(file_path, window)
                count_x += 1
        count_y += 1


def process_images(img_files, img_dir, crop_dir, window_size, desired_overlap):
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    for img_file in img_files:
        img_name = img_file.split(".")[0]
        img_path = os.path.join(img_dir, img_file)
        generate_sliding_window_images(img_path, crop_dir, window_size, desired_overlap)


def crop(path, crop_path, window_size, desired_overlap):
    classname_list = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']
    for ct in classname_list:
        print(f"{ct} processing...")
        if not os.path.isdir(os.path.join(path, ct)):
            continue
        ct_path = os.path.join(path, ct)
        cp_path = os.path.join(crop_path, ct)
        for category in os.listdir(ct_path):
            category_path = os.path.join(ct_path, category)
            crop_path_1 = os.path.join(cp_path, category)
            if category in ['test_private', 'test_private_mixed']:
                img_files = os.listdir(category_path)
                process_images(img_files, category_path, crop_path_1, window_size, desired_overlap)
            else:
                for label in os.listdir(category_path):
                    label_path = os.path.join(category_path, label)
                    crop_path_2 = os.path.join(crop_path_1, label)

                    if label == 'ground_truth':
                        for gt in os.listdir(label_path):
                            gt_path = os.path.join(label_path, gt)
                            crop_path_3 = os.path.join(crop_path_2, gt)
                            img_files = os.listdir(gt_path)
                            process_images(img_files, gt_path, crop_path_3, window_size, desired_overlap)
                    else:
                        img_files = os.listdir(label_path)
                        process_images(img_files, label_path, crop_path_2, window_size, desired_overlap)
        print(f"{ct} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'./data/mvtec_ad_2')
    parser.add_argument('--save_path', type=str, default=r'./mvtec_ad_2_processed')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    crop(path=args.data_path, crop_path=args.save_path, window_size=1024, desired_overlap=0.1)

