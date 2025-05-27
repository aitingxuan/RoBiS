# bash VAND2025_track1_MAD2_reproduce_final_result.sh
# The dataset path needs to be set manually
dataset_dir='./data/mvtec_ad_2'
device=0

# Pre-processing steps (about 50 min)
processed_dir='./mvtec_ad_2_processed'
python swin-cropping.py --data_path ${dataset_dir} --save_path ${processed_dir}

# Train call: please reserve at least 17GB of GPU memory
# You can use different GPUs to train different categories to reduce time consumption
# You can also download the trained checkpoints by the below link:
# https://drive.google.com/drive/folders/1JvbEru6W1RxThjjiPJSONbO97j9_I6dN?usp=drive_link
CUDA_VISIBLE_DEVICES=${device} python INP_Former_Single_Class.py \
--data_path ${processed_dir} --save_dir ./saved_weights --phase train \
--mvtecad2_class_list sheet_metal vial wallplugs walnuts can fabric fruit_jelly rice

# Test call:
amap_savedir='./anomaly_map_results'
CUDA_VISIBLE_DEVICES=${device} python INP_Former_Single_Class.py \
--data_path ${processed_dir} --save_dir ./saved_weights --amap_savedir ${amap_savedir} --phase test --test_type test_private \
--mvtecad2_class_list sheet_metal vial wallplugs walnuts can fabric fruit_jelly rice

CUDA_VISIBLE_DEVICES=${device} python INP_Former_Single_Class.py \
--data_path ${processed_dir} --save_dir ./saved_weights --amap_savedir ${amap_savedir} --phase test --test_type test_private_mixed \
--mvtecad2_class_list sheet_metal vial wallplugs walnuts can fabric fruit_jelly rice

# Post-processing steps
python merging.py --amap_savedir ${amap_savedir} --test_type challenge

# Binarization
bin_savedir='./binary_map_results'
python binarization.py --amap_savedir ${amap_savedir} --bin_savedir ${bin_savedir} --test_type challenge
CUDA_VISIBLE_DEVICES=${device} python SAM-Finer.py --data_path ${dataset_dir} --bin_savedir ${bin_savedir} --test_type challenge

# Expected directory for evaluation
mkdir -p ./submission_folder
cp -r ${amap_savedir}/anomaly_images ./submission_folder/
cp -r ${bin_savedir}/anomaly_images_thresholded ./submission_folder/

