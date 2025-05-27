import random
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')

def adjust_exposure(image, exposure_value=None):
    # (H, W, 3)
    image = np.float32(image)
    image = image * (2 ** exposure_value)
    image = np.clip(image, 0, 255)
    # noise add
    mean, std = 0, 15
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    image = Image.fromarray(image)
    return image

class RandomExposureAdjustment:
    def __init__(self, overexposure_factor=1.0, underexposure_factor=-1.0):
        self.overexposure_factor = overexposure_factor
        self.underexposure_factor = underexposure_factor

    def __call__(self, image):
        if random.random() > 0.5:
            image = adjust_exposure(image, random.uniform(-0.2, 0.2))
        return image
        
def get_data_transforms(size, isize, mean_train=None, std_train=None, bright_aug=False):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    if bright_aug:
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            RandomExposureAdjustment(overexposure_factor=0.5, underexposure_factor=-0.5),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTec2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, test_type):
        self.test_type = test_type
        self.phase = phase
        if phase == 'train':
            self.img_path = root
        else:
            test_data_type = test_type
            self.img_path = os.path.join(root, test_data_type)
            self.gt_path = os.path.join(root, test_data_type, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0
        self.bright_tag = False
        self.resize_shape = (518, 518)

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if self.test_type == 'test_public':
            defect_types = os.listdir(self.img_path)
            for defect_type in defect_types:
                if defect_type == 'good':
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                                glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                                glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
                else:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                                glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                                glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))
        else:
            if self.phase == 'train':
                img_tot_paths = os.listdir(os.path.join(self.img_path, 'good'))
                img_tot_paths = [os.path.join(self.img_path, 'good', x) for x in img_tot_paths]
            else:
                img_tot_paths = os.listdir(self.img_path)
                img_tot_paths = [os.path.join(self.img_path, x) for x in img_tot_paths]
            img_tot_paths.sort()
            gt_tot_paths.extend([0] * len(img_tot_paths))
            tot_labels.extend([0] * len(img_tot_paths))
            tot_types.extend(['good'] * len(img_tot_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
            if gt.shape[0] == 3:
                gt = gt[0:1, :, :]
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path

