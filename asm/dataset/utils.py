import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    boxes[:, 0] = boxes[:, 0] / current_dim * orig_w
    boxes[:, 1] = boxes[:, 1] / current_dim * orig_h  # y1
    boxes[:, 2] = boxes[:, 2] / current_dim * orig_w
    boxes[:, 3] = boxes[:, 3] / current_dim * orig_h
    return boxes


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize(image, size=416):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def change_size_for_yolo(img, size_w=416, size_h=416):
    # img = Image.fromarray(img)
    img = img.resize((size_w, size_h), Image.ANTIALIAS)
    img = transforms.ToTensor()(img)

    img, _ = pad_to_square(img, 0)
    img = resize(img, size_w)
    return img


def cvt_xywh_2pts(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
