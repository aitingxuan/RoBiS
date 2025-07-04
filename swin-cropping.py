import os
import cv2
import numpy as np
import argparse

def generate_sliding_window_images(img_path, img_crop_path, window_size, desired_overlap):
    window_size = window_size
    desired_overlap = desired_overlap
    step_size = int(window_size * (1 - desired_overlap))
    
    img_name = os.path.basename(img_path).split(".")[0]
    img = cv2.imread(img_path)
    
    height, width, _ = img.shape
    
    num_steps_y = int(np.ceil((height - window_size) / step_size)) + 1
    num_steps_x = int(np.ceil((width - window_size) / step_size)) + 1
    
    y_steps = [i * step_size for i in range(num_steps_y)]
    x_steps = [i * step_size for i in range(num_steps_x)]
    
    count_y = 0
    for y in y_steps:
        count_x = 0
        for x in x_steps:
            window_x1 = x
            window_y1 = y
            window_x2 = x + window_size
            window_y2 = y + window_size
            window = img[window_y1:window_y2, window_x1:window_x2]
            
            if window.shape[0] < window_size or window.shape[1] < window_size:
                padded_window = np.zeros((window_size, window_size, 3), dtype=np.uint8)
                padded_window[:window.shape[0], :window.shape[1]] = window
                window = padded_window

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

