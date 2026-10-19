"""三种图像增强：只改变训练输入，不改变标签。使用 torch 随机数以便复现。"""
import torch


class RandomFlip:
    """输入 [C,H,W]，分别以 p 概率左右/上下翻转，输出同形状图像。"""
    def __init__(self, p=0.5):
        if not 0 <= p <= 1:
            raise ValueError(f'RandomFlip p expected [0,1], got {p}')
        self.p = p

    def __call__(self, image):
        if torch.rand(()) < self.p:
            image = image.flip(-1)
        if torch.rand(()) < self.p:
            image = image.flip(-2)
        return image


class GaussianNoise:
    """以 p 概率添加零均值噪声，sigma 在区间随机取值，输出裁剪至 [0,1]。"""
    def __init__(self, p=0.5, sigma=(0.01, 0.08)):
        if not 0 <= p <= 1 or len(sigma) != 2 or not 0 <= sigma[0] <= sigma[1]:
            raise ValueError(f'GaussianNoise invalid p={p}, sigma={sigma}')
        self.p, self.sigma = p, sigma

    def __call__(self, image):
        if torch.rand(()) >= self.p:
            return image
        sigma = torch.empty(()).uniform_(*self.sigma)
        return (image + torch.randn_like(image) * sigma).clamp(0, 1)


class RandomContrast:
    """围绕图像均值调整对比度，以 p 概率应用，输出仍在 [0,1]。"""
    def __init__(self, p=0.5, factor=(0.65, 1.15)):
        if not 0 <= p <= 1 or len(factor) != 2 or not 0 <= factor[0] <= factor[1]:
            raise ValueError(f'RandomContrast invalid p={p}, factor={factor}')
        self.p, self.factor = p, factor

    def __call__(self, image):
        if torch.rand(()) >= self.p:
            return image
        factor = torch.empty(()).uniform_(*self.factor)
        return (image.mean() + factor * (image - image.mean())).clamp(0, 1)


# 新增增强：实现 __call__(image)，在此登记，再通过 JSON 配置启用。
TRANSFORMS = {'flip': RandomFlip, 'noise': GaussianNoise, 'contrast': RandomContrast}


class Compose:
    """按配置列表顺序应用增强；多个增强可在同一张图上叠加。"""
    def __init__(self, settings):
        self.transforms = []
        for setting in settings:
            options = dict(setting)
            name = options.pop('name')
            if name not in TRANSFORMS:
                raise ValueError(f'Unknown augmentation {name!r}; available={list(TRANSFORMS)}')
            self.transforms.append(TRANSFORMS[name](**options))

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image
