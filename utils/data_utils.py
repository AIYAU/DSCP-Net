"""数据处理工具"""
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import json


def get_dataset_info(dataset_name):
    """获取数据集配置信息"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'dataset_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config[dataset_name]


def padding_data(data, patch_size):
    """边界填充"""
    distance = patch_size // 2
    C, H, W = data.shape
    padded = torch.zeros(C, H + 2 * distance, W + 2 * distance)
    padded[:, distance:distance+H, distance:distance+W] = data
    # 镜像填充边界
    padded[:, :distance, :] = padded[:, distance:2*distance, :].flip(1)
    padded[:, -distance:, :] = padded[:, -2*distance:-distance, :].flip(1)
    padded[:, :, :distance] = padded[:, :, distance:2*distance].flip(2)
    padded[:, :, -distance:] = padded[:, :, -2*distance:-distance].flip(2)
    return padded


def transform_gt(gt, known_classes, unknown_classes):
    """转换标签：已知类重编号为1~N，未知类为N+1"""
    gt_new = np.zeros_like(gt)
    for cls in unknown_classes:
        gt_new[gt == cls] = len(known_classes) + 1
    for idx, cls in enumerate(known_classes):
        gt_new[gt == cls] = idx + 1
    return gt_new


def split_data(gt, known_classes, train_num, seed, val_ratio=0.2):
    """划分训练集/验证集/测试集"""
    train_idx, val_idx, test_idx = [], [], []

    for cls in range(1, len(known_classes) + 1):
        x, y = np.where(gt == cls)
        locations = x * gt.shape[1] + y
        n_samples = len(locations)

        if train_num >= n_samples:
            train_num_cls = max(2, n_samples // 2)
        else:
            train_num_cls = train_num

        train_val, test = train_test_split(locations, train_size=train_num_cls, random_state=seed)
        if len(train_val) <= 1 or val_ratio <= 0:
            train = train_val
            val = np.array([], dtype=train_val.dtype)
        else:
            val_num = max(1, int(round(len(train_val) * val_ratio)))
            val_num = min(val_num, len(train_val) - 1)
            train, val = train_test_split(train_val, test_size=val_num, random_state=seed)

        train_idx.extend(train.tolist())
        val_idx.extend(val.tolist())
        test_idx.extend(test.tolist())
        print(f'Class {cls}: total {n_samples}, train {len(train)}, val {len(val)}, test {len(test)}')

    x, y = np.where(gt == len(known_classes) + 1)
    unknown_idx = (x * gt.shape[1] + y).tolist()

    print(f'Total: train {len(train_idx)}, val {len(val_idx)}, known_test {len(test_idx)}, unknown_test {len(unknown_idx)}')

    return train_idx, val_idx, test_idx, unknown_idx


class BaseDataset(Dataset):
    """基础数据集类"""
    def __init__(self, data, patch_size):
        self.data = data
        self.patch_size = patch_size
        self.H = data.shape[1] - patch_size + 1
        self.W = data.shape[2] - patch_size + 1

    def get_patch(self, idx):
        x = idx // self.W
        y = idx % self.W
        return self.data[:, x:x+self.patch_size, y:y+self.patch_size]

    def __len__(self):
        return self.H * self.W


class LabelDataset(BaseDataset):
    """带标签的数据集"""
    def __init__(self, data, gt, patch_size, indices):
        super().__init__(data, patch_size)
        self.gt = gt
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        loc = self.indices[idx]
        patch = self.get_patch(loc)
        x = loc // self.W
        y = loc % self.W
        label = self.gt[x, y] - 1
        return patch, label, loc


class AllDataset(BaseDataset):
    """全图预测数据集"""
    def __init__(self, data, patch_size):
        super().__init__(data, patch_size)

    def __getitem__(self, idx):
        return self.get_patch(idx)


def init_dataset(args):
    """初始化数据集"""
    info = get_dataset_info(args.dataset)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', info['path'])

    data = loadmat(os.path.join(data_dir, info['file_name']))[info['mat_name']].astype(np.float32)
    data = torch.from_numpy(data).permute(2, 0, 1)
    data = (data - data.min()) / (data.max() - data.min())
    data = padding_data(data, args.patch)

    gt = loadmat(os.path.join(data_dir, info['gt_file_name']))[info['gt_mat_name']].astype(np.int64)
    gt = transform_gt(gt, args.known_classes, args.unknown_classes)

    train_idx, val_idx, test_idx, unknown_idx = split_data(
        gt, args.known_classes, args.train_num, args.seed, args.val_ratio
    )

    return {
        'data': data,
        'gt': torch.from_numpy(gt),
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'unknown_idx': unknown_idx,
        'num_classes': len(args.known_classes),
        'num_bands': info['bands_num'],
        'patch_size': args.patch,
        'name': args.dataset,
        'path': info['path'],
        'gt_file_name': info['gt_file_name'],
        'gt_mat_name': info['gt_mat_name'],
        'known_classes': args.known_classes,
        'unknown_classes': args.unknown_classes
    }


def get_dataloader(dataset_info, batch_size, mode='train'):
    """获取数据加载器"""
    data = dataset_info['data']
    gt = dataset_info['gt']
    patch_size = dataset_info.get('patch_size', 9)

    if mode == 'train':
        dataset = LabelDataset(data, gt, patch_size, dataset_info['train_idx'])
    elif mode == 'val':
        dataset = LabelDataset(data, gt, patch_size, dataset_info['val_idx'])
    elif mode == 'test':
        test_idx = dataset_info['test_idx'] + dataset_info['unknown_idx']
        dataset = LabelDataset(data, gt, patch_size, test_idx)
    elif mode == 'all':
        dataset = AllDataset(data, patch_size)
    else:
        raise ValueError(f'Unsupported dataloader mode: {mode}')

    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=4)
