import argparse
import yaml
import os
import json
import shutil
import cv2
from tqdm import tqdm
import csv
import sys
sys.path.append(os.getcwd())
from MEBin import MEBin


mvtec_class_names = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    
def mvtec_bin(args, test_type='test_public'):
    '''
    Binarize anomaly maps for the MVTec dataset and save the results.
    This function processes each class in the MVTec dataset, binarizes the anomaly maps using the MEBin algorithm,
    and saves the binarized maps to the specified output path.
    
    Args:
        args (dict): Configuration arguments
    Returns:
        None
    '''
    anomaly_map_dir = args.amap_savedir
    binarization_dir = args.bin_savedir
    if test_type == 'test_public':
        anomaly_map_path = os.path.join(anomaly_map_dir, 'anomaly_images_public')
        output_path = os.path.join(binarization_dir, 'anomaly_images_thresholded_public')
    else:
        anomaly_map_path = os.path.join(anomaly_map_dir, 'anomaly_images')
        output_path = os.path.join(binarization_dir, 'anomaly_images_thresholded')

    for class_name in mvtec_class_names:
        print(f'Binarizing {class_name}...')
        class_output_path = os.path.join(output_path, class_name, test_type)
        os.makedirs(class_output_path, exist_ok=True)
        
        # Collect anomaly map paths
        anomaly_map_path_list = []
        anomaly_num_list = []
        anomaly_types = sorted(os.listdir(os.path.join(anomaly_map_path, class_name, test_type)))
        
        if test_type == 'test_public':
            for anomaly_type in anomaly_types:
                anomaly_type_anomaly_map_paths = sorted(os.listdir(os.path.join(anomaly_map_path, class_name, test_type, anomaly_type)))
                anomaly_map_path_list.extend([os.path.join(anomaly_map_path, class_name, test_type, anomaly_type, path) for path in anomaly_type_anomaly_map_paths])
                anomaly_num = len(anomaly_type_anomaly_map_paths)
                anomaly_num_list.append(anomaly_num)
        else:
            anomaly_types = ['unknown']
            anomaly_type_anomaly_map_paths = sorted(os.listdir(os.path.join(anomaly_map_path, class_name, test_type)))
            anomaly_map_path_list.extend([os.path.join(anomaly_map_path, class_name, test_type, path) for path in anomaly_type_anomaly_map_paths])
            anomaly_num = len(anomaly_type_anomaly_map_paths)
            anomaly_num_list.append(anomaly_num)
        
        # instantiate the binarization method
        bin = MEBin(anomaly_map_path_list, class_name=class_name)
        
        # Use the selected binarization method to binarize the anomaly maps
        binarized_maps, threshold_list = bin.binarize_anomaly_maps()
        
        # Save the binarization result
        start = 0
        class_threshold_dict = {}
        for i, anomaly_type in enumerate(anomaly_types):
            if test_type == 'test_public':
                anomaly_type_out_path = os.path.join(class_output_path, anomaly_type)
            else:
                anomaly_type_out_path = os.path.join(class_output_path)
            os.makedirs(anomaly_type_out_path, exist_ok=True)
            end = start + anomaly_num_list[i]
            anomaly_type_binarized_maps = binarized_maps[start:end]
            anomaly_type_thresholds = threshold_list[start:end]
            
            # Iterate over the binarized maps and thresholds for the current anomaly type
            class_threshold_dict[anomaly_type] = {}
            for j, threshold in enumerate(anomaly_type_thresholds):
                map_name = os.path.basename(anomaly_map_path_list[start + j])
                class_threshold_dict[anomaly_type][map_name] = threshold
            
            # Save the binarized maps for the current anomaly type
            for j, binarized_map in enumerate(anomaly_type_binarized_maps):
                map_path = os.path.join(anomaly_type_out_path, os.path.basename(anomaly_map_path_list[start + j]))
                map_path = map_path.replace('.tiff', '.png')
                cv2.imwrite(map_path, binarized_map)

            start = end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--amap_savedir', type=str, default=r'./anomaly_map_results')
    parser.add_argument('--bin_savedir', type=str, default=r'./binary_map_results')
    parser.add_argument('--test_type', type=str, default=r'challenge')
    args = parser.parse_args()
    if args.test_type != 'challenge':
        mvtec_bin(args, test_type='test_public')
    else:
        mvtec_bin(args, test_type='test_private')
        mvtec_bin(args, test_type='test_private_mixed')
