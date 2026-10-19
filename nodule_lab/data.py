"""官方 NPZ 数据读取；训练、验证、测试使用原始划分。"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

MD5 = '8755a7e9e05a4d9ce80a24c3e7a256f3'
URL = 'https://zenodo.org/records/10519652/files/nodulemnist3d.npz?download=1'


class NoduleDataset(Dataset):
    """读取一个划分，返回图像 float32 [1,28,28]、标签 float32 标量。"""
    def __init__(self, path, split, transform=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'NoduleMNIST3D split={split!r}; expected train/val/test')
        if transform is not None and split != 'train':
            raise ValueError(f'NoduleMNIST3D split={split}: augmentation is train-only')
        self.path, self.split, self.transform = Path(path), split, transform
        with np.load(self.path, allow_pickle=False) as data:
            images = data[f'{split}_images']
            labels = data[f'{split}_labels']
        context = f'NoduleMNIST3D {self.path} {split}'
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (28, 28, 28):
            raise ValueError(f'{context}_images expected uint8 [N,28,28,28], got {images.dtype} {images.shape}')
        if len(images) == 0 or labels.shape != (len(images), 1):
            raise ValueError(f'{context}_labels expected nonempty [N,1], got {labels.shape}, N={len(images)}')
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f'{context}_labels expected 0/1, got {np.unique(labels)}')
        # 归一化只做一次；保留中心切片，不把不同体积重新划分。
        self.images = torch.from_numpy(images[:, 14:15].astype(np.float32) / 255.)
        self.labels = torch.from_numpy(labels[:, 0].astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 克隆后增强，避免原地操作污染下一轮读取的原始图像。
        image = self.images[index].clone()
        if self.transform is not None:
            image = self.transform(image)
        if image.shape != (1, 28, 28) or image.dtype != torch.float32:
            raise ValueError(f'{self.path} {self.split}[{index}]: transform expected float32 [1,28,28], got {image.dtype} {image.shape}')
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f'{self.path} {self.split}[{index}]: transform output must be finite in [0,1]')
        return image, self.labels[index].clone()
