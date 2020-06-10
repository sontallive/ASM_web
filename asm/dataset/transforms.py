import random
from torchvision.transforms import functional as F
import torch
import numpy as np
from PIL import Image, ImageEnhance


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def vertical_flip(images, targets):
    images = torch.flip(images, [-2])
    targets[:, 3] = 1 - targets[:, 3]
    return images, targets


def random_enhance(img):
    if np.random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(0.5 + np.random.random())
    if np.random.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(0.5 + np.random.random())
    if np.random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(0.5 + np.random.random())
    return img


class Compose(object):
    """
    as sample (image, target), not use torchvision.transforms.Compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:  # a bunch of transforms
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class RandomEnhance(object):
    def __call__(self, img, target):
        img = random_enhance(img)
        return img, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)  # use torchvision func to_tensor
        return image, target
