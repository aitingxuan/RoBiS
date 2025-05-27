# ✨RoBiS: Robust Binary Segmentation for High-Resolution Industrial Images (VAND3.0 challenge)

Name(s):
[Xurui Li](https://github.com/xrli-U)<sup>1</sup> | [Zhongsheng Jiang](https://github.com/FoundWind7)<sup>1</sup> | [Tingxuan Ai](https://aitingxuan.github.io/)<sup>1</sup> | [Yu Zhou](https://github.com/zhouyu-hust)<sup>1,2</sup>

Affiliation(s):
<sup>1</sup>Huazhong University of Science and Technology | <sup>2</sup>Wuhan JingCe Electronic Group Co.,LTD

Contact Information:
**xrli\_plus@hust.edu.cn** | zsjiang@hust.edu.cn | tingxuanai@hust.edu.cn | yuzhou@hust.edu.cn

Track: Adapt \& Detect---Robust Anomaly Detection in Real-World Applications

### Technical report: [ResearchGate](https://www.researchgate.net/publication/392124350_RoBiS_Robust_Binary_Segmentation_for_High-Resolution_Industrial_Images) | [PDF](RoBiS.pdf)


## 🧐Overview

This repository is the official implementation of our solution **RoBiS** for the CVPR2025 VAND3.0 challenge Track 1.

MVTec Benchmark Server: [https://benchmark.mvtec.com/](https://benchmark.mvtec.com/).

Challenge Website: [https://sites.google.com/view/vand30cvpr2025/challenge](https://sites.google.com/view/vand30cvpr2025/challenge)


## 🎯Setup

### Environment:

- Python 3.8
- CUDA 11.7
- PyTorch 2.0.1

Clone the repository locally:

```
git clone https://github.com/xrli-U/RoBiS.git
```

Create virtual environment:

```
conda create --name RoBiS python=3.8
conda activate RoBiS
```

Install the required packages:

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```
## 👇Prepare Datasets

Download the [MVTec AD 2](https://arxiv.org/pdf/2503.21622) dataset through the official link ([web](https://www.mvtec.com/company/research/datasets/mvtec-ad-2))

Put the datasets in `./data` folder.

```
data
|---mvtec_ad_2
|-----|-- can
|-----|-----|----- test_private
|-----|-----|----- test_private_mixed
|-----|-----|----- test_public
|-----|-----|----- train
|-----|-----|----- validation
|-----|-- fabric
|-----|--- ...
```

## 💎Run RoBiS
Before starting to run our RoBiS, execute the `download_weights.sh` script to download the pre-training weights.
```
bash download_weights.sh
```
We provide two ways to run our code.

### One bash for all operations
```
bash VAND2025_track1_MAD2_reproduce_final_result.sh
```
You can run the above script for all steps in our method.
The continuous anomaly maps and thresholded binary masks are stored in `./submission_folder` for evaluation.
The final continuous anomaly maps could be download in [google drive](https://drive.google.com/file/d/1OqejveTgEuYr9obEUV3h3Vzq2HTp29ua/view?usp=sharing).
The final thresholded binary masks could be download in [google drive](https://drive.google.com/file/d/1ilMnxisuQOYnvllu1kUHaibkzHiHN_R-/view?usp=sharing).
For more detailed arguments of this script, please refer to the following *step by step*.


### Step by step
**1. Pre-processing**
```
python swin-cropping.py --data_path ./data/mvtec_ad_2 --save_path ./mvtec_ad_2_processed
```
Please leave about 50GB for the pre-processed data.

Key arguments:
- `--data_path`: The directory of the original dataset.
- `--save_path`: The directory that saves the pre-processed dataset. This directory will be automatically created.

**2. Model training**
```
CUDA_VISIBLE_DEVICES=0 python INP_Former_Single_Class.py \
--data_path ./mvtec_ad_2_processed --save_dir ./saved_weights --phase train \
--mvtecad2_class_list sheet_metal vial wallplugs walnuts can fabric fruit_jelly rice
```
We use ViT-B-14 initialized with DINOv2-R pre-trained weights as the encoder.
The pre-trained weights will be download automatically as `./backbones/weights/dinov2_vitb14_reg4_pretrain.pth`

To train the AD model under the default settings, please reserve at least 17GB of GPU memory.
You can use different GPUs to train different categories to reduce time consumption.
You can also download the trained checkpoints by this link [(google drive)](https://drive.google.com/drive/folders/1JvbEru6W1RxThjjiPJSONbO97j9_I6dN?usp=drive_link).

Key arguments:
- `--data_path`: The directory of the pre-processed dataset.
- `--save_dir`: The directory that saves model weights. This directory will be automatically created.
- `mvtecad2_class_list`: The product categories of MVTec AD 2 dataset. Since our method trains one model for each category, different GPUs could be used to train different categories.

**3. Model testing**
```
# Testing for test_private
CUDA_VISIBLE_DEVICES=0 python INP_Former_Single_Class.py \
--data_path ./mvtec_ad_2_processed --save_dir ./saved_weights --amap_savedir ./anomaly_map_results --phase test --test_type test_private \
--mvtecad2_class_list sheet_metal vial wallplugs walnuts can fabric fruit_jelly rice

# Testing for test_private_mixed
CUDA_VISIBLE_DEVICES=0 python INP_Former_Single_Class.py \
--data_path ./mvtec_ad_2_processed --save_dir ./saved_weights --amap_savedir ./anomaly_map_results --phase test --test_type test_private_mixed \
--mvtecad2_class_list sheet_metal vial wallplugs walnuts can fabric fruit_jelly rice
```
Key arguments:
- `--data_path`: The directory of the pre-processed dataset.
- `--save_dir`: The directory that saves model weights.
- `--amap_savedir`: The directory that saves anomaly maps *(.tiff)* of all sub-images. This directory will be automatically created.
- `--test_type`: The test set of MVTec AD 2 dataset, setting *test_private* or *test_private_mixed*.
- `mvtecad2_class_list`: The product categories of MVTec AD 2 dataset.

**4. Post-processing**
```
python merging.py --amap_savedir ./anomaly_map_results --test_type challenge
```
Merging the anomaly maps of sub-images into the corresponding original anomaly map.

Key arguments:
- `--amap_savedir`: The directory that saves anomaly maps *(.tiff)* of all sub-images. After the merge, the anomaly maps of sub-images are automatically deleted.

**5. Binarization**
```
# Using MEBin and mean+3std to generate coarse binary masks.
python binarization.py --amap_savedir ./anomaly_map_results --bin_savedir ./binary_map_results --test_type challenge

# Using SAM to generate finer binary masks.
CUDA_VISIBLE_DEVICES=0 python SAM-Finer.py --data_path ./data/mvtec_ad_2 --bin_savedir ./binary_map_results --test_type challenge
```
Before running `SAM-Finer.py`, make sure that the pre-trained weights *(sam_b and sam_h)* of SAM are downloaded to the current directory (`bash download_weights.sh`).

Key arguments:
- `--amap_savedir`: The directory that saves anomaly maps *(.tiff)*.
- `--bin_savedir`: The directory that saves thresholded binary masks.
- `--data_path`: The directory of the original dataset.

**6. Zip for evaluation**
We transfer the continuous anomaly maps and thresholded binary masks to `./submission_folder` for compression and evaluation.
```
mkdir -p ./submission_folder
cp -r ${amap_savedir}/anomaly_images ./submission_folder/
cp -r ${bin_savedir}/anomaly_images_thresholded ./submission_folder/
```
The final continuous anomaly maps could be download in [google drive](https://drive.google.com/file/d/1OqejveTgEuYr9obEUV3h3Vzq2HTp29ua/view?usp=sharing).

The final thresholded binary masks could be download in [google drive](https://drive.google.com/file/d/1ilMnxisuQOYnvllu1kUHaibkzHiHN_R-/view?usp=sharing).

## 🎖️Results

All the results are calculated by the official leaderboard.

### MVTec AD 2

|   Object    | AucPro_0.05 |  ClassF1  |   SegF1   |   AucPro_0.05   |     ClassF1     |      SegF1      |
| :---------: | :---------: | :-------: | :-------: | :-------------: | :-------------: | :-------------: |
|             |  (private)  | (private) | (private) | (private_mixed) | (private_mixed) | (private_mixed) |
|     Can     |    30.28    |   60.93   |   1.86    |      20.03      |      65.04      |      0.84       |
|   Fabric    |    79.45    |   83.79   |   87.46   |      79.27      |      83.80      |      73.37      |
| Fruit Jelly |    74.46    |   87.35   |   53.63   |      74.11      |      87.55      |      52.62      |
|    Rice     |    62.27    |   72.00   |   63.86   |      63.89      |      73.45      |      63.23      |
| Sheet Metal |    75.51    |   87.68   |   70.98   |      73.54      |      86.69      |      70.92      |
|    Vial     |    76.81    |   84.61   |   48.73   |      69.59      |      85.77      |      48.83      |
| Wall Plugs  |    62.20    |   75.20   |   14.38   |      24.77      |      72.66      |      3.40       |
|   Walnuts   |    77.05    |   85.42   |   67.13   |      72.00      |      83.95      |      58.94      |
|    Mean     |    67.25    |   79.62   |   51.00   |      59.65      |      79.86      |      46.52      |


## Thanks

Our repo is built on [INP-Former](https://github.com/luow23/INP-Former), thanks their clear and elegant code !

## License
RoBiS is released under the **MIT Licence**.

