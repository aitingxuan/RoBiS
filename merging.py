import os
import shutil
import cv2
import numpy as np
import math
import tifffile as tiff
import argparse
from tqdm import tqdm

resolution_dict = {'sheet_metal':(4224, 1056), 'vial':(1400, 1900), 'wallplugs':(2448, 2048), 'walnuts':(2448, 2048),
                    'can':(2232, 1024), 'fabric':(2448, 2048), 'fruit_jelly':(2100, 1520), 'rice':(2448, 2048)}

def reconstruct_image(crop_path, ct, window_size, desired_overlap):
    step_size = int(window_size * (1 - desired_overlap))
    crop_files = sorted([f for f in os.listdir(crop_path) if f.endswith(".tiff")])

    original_width, original_height = resolution_dict[ct]
    original_width = int(original_width/4)
    original_height = int(original_height/4)
    
    grouped_files = {}
    for crop_file in crop_files:
        prefix = "_".join(crop_file.split("_")[:-1])
        if prefix not in grouped_files:
            grouped_files[prefix] = []
        grouped_files[prefix].append(crop_file)

    for prefix, files in tqdm(grouped_files.items()):
        padded_height = math.ceil(original_height / window_size) * window_size
        padded_width = math.ceil(original_width / window_size) * window_size

        reconstructed_image = np.zeros((padded_height, padded_width), dtype=np.float16)
        weight_matrix = np.zeros((padded_height, padded_width), dtype=np.float16)

        for crop_file in files:
            parts1 = crop_file.split(".")
            parts = parts1[0].split("_")
            index = [int(a) for a in str(parts[-1])]
            row_idx = int(index[0])
            col_idx = int(index[1])
            y1 = row_idx * step_size
            x1 = col_idx * step_size
            y2 = y1 + window_size
            x2 = x1 + window_size

            crop_img = tiff.imread(os.path.join(crop_path, crop_file))
            h, w = crop_img.shape

            if y2 > padded_height:
                h = padded_height - y1
                y2 = padded_height
            if x2 > padded_width:
                w = padded_width - x1
                x2 = padded_width
                crop_img = crop_img[:h, :w]

            reconstructed_image[y1:y2, x1:x2] += crop_img[:h, :w]
            weight_matrix[y1:y2, x1:x2] += 1
            os.remove(os.path.join(crop_path, crop_file))

        weight_matrix[weight_matrix == 0] = 1
        reconstructed_image = (reconstructed_image / weight_matrix[:, :]).astype(np.float16)
        reconstructed_image = reconstructed_image[:original_height, :original_width]
        save_path = os.path.join(crop_path, f"{prefix}.tiff")
        tiff.imwrite(save_path, reconstructed_image)


def merge(crop_path, window_size=256, desired_overlap=0.1):
    classname_list = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']
    for classname in classname_list:
        print(f"{classname} processing...")
        for test_type in os.listdir(os.path.join(crop_path,classname)):
            crop_path_1 = os.path.join(crop_path,classname,test_type)
            if test_type == 'test_private' or test_type =='test_private_mixed':
                reconstruct_image(crop_path_1, classname, window_size, desired_overlap)
            else:
                for label in os.listdir(crop_path_1):
                    crop_path_2 = os.path.join(crop_path_1,label)
                    reconstruct_image(crop_path_2, classname, window_size, desired_overlap)
        print(f"{classname} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--amap_savedir', type=str, default=r'./anomaly_map_results')
    parser.add_argument('--test_type', type=str, default=r'challenge')
    args = parser.parse_args()
    if args.test_type != 'challenge':
        crop_path = os.path.join(args.amap_savedir, 'anomaly_images_public')
    else:
        crop_path = os.path.join(args.amap_savedir, 'anomaly_images')
    merge(crop_path=crop_path, window_size=256, desired_overlap=0.1)