import os
import sys
import glob
import h5py
import numpy as np
from sklearn import preprocessing
from torch.utils import data
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image
from .plyfile import load_ply
from . import data_utils as d_utils
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_DIR = 'data/ShapeNetRendering'
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 产生不同的version
trans_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])
    
trans_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

def load_modelnet_data(partition):
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_ScanObjectNN(partition):
    BASE_DIR = 'data/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    
    return data, label

def load_shapenet_data():
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_filepath = []

    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs
        
    return all_filepath

        
class ShapeNet(Dataset):
    def __init__(self):
        self.data = load_shapenet_data()

    
    def __getitem__(self, item):
        pcd_path = self.data[item]
        pointcloud_1 = load_ply(self.data[item])
        pointcloud_2 = load_ply(self.data[item])
        point_t1 = trans_1(pointcloud_1)
        point_t2 = trans_2(pointcloud_2)

        pointcloud = (point_t1, point_t2)
        return pointcloud

    def __len__(self):
        return len(self.data)



class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


