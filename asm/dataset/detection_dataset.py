import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
import asm.dataset.transforms as T
from asm.dataset.transforms import random_enhance
from torchvision.transforms import ToTensor
from asm.dataset.utils import cvt_xywh_2pts, pad_to_square, resize
from PIL import Image
import random


class Detection_Dataset_rcnn(Dataset):
    def __init__(self,
                 root,
                 coco_json,
                 transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file=os.path.join(root, coco_json))
        self.imgs, self.img_anns = [], []
        for img_id, anns in self.coco.imgToAnns.items():
            self.imgs.append(self.coco.imgs[img_id]['file_name'])  # img_path
            self.img_anns.append(anns)

    def __getitem__(self, idx):
        img_anns = self.img_anns[idx]  # list
        img_path = os.path.join(self.imgs[idx])

        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        for ann in img_anns:
            boxes.append(cvt_xywh_2pts(ann['bbox']))
            labels.append(ann['category_id'] + 1)  # bg=0, infer will del, so cigar_A can't show

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # cal area by box, return vector
        iscrowd = torch.zeros((len(img_anns),), dtype=torch.int64)  # set 0

        target = {
            'image_id': torch.as_tensor([idx]),
            "boxes": boxes,  # [x0, y0, x1, y1] ~ [0,W], [0,H]
            "labels": labels,  # class label
            "area": area,  # used in COCO metric, AP_small,medium,large
            "iscrowd": iscrowd,  # if True, ignored during eval
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Detection_Dataset_yolo(Dataset):
    def __init__(self,
                 root,  # image root
                 coco_json,  # coco format annotation
                 img_size_w, img_size_h,
                 augment=True,  # tricks
                 multiscale=True,
                 normalized_labels=True,
                 mixup=False,
                 aug_double=False):
        self.root = root
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h
        # multiscale image size range [min_size, max_size]
        self.min_size = max(self.img_size_w, self.img_size_h) - 3 * 32
        self.max_size = max(self.img_size_w, self.img_size_h) + 3 * 32
        # tricks
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.mixup = mixup
        self.aug_double = aug_double
        self.batch_count = 0  # for multiscale each 10 batchs
        self.coco = COCO(annotation_file=os.path.join(root, coco_json))
        self.img_size = max(self.img_size_h, self.img_size_w)
        self.img_files = []
        self.img_anns = []
        self.is_hor = []
        for img_id, anns in self.coco.imgToAnns.items():
            self.img_files.append(self.coco.imgs[img_id]['file_name'])
            self.img_anns.append(anns)
            self.is_hor.append(0)
            if aug_double:  # add 3 more, augment in different way, H,V,HV
                self.img_files.append(self.coco.imgs[img_id]['file_name'])
                self.img_anns.append(anns)
                self.is_hor.append(1)
                self.img_files.append(self.coco.imgs[img_id]['file_name'])
                self.img_anns.append(anns)
                self.is_hor.append(2)
                self.img_files.append(self.coco.imgs[img_id]['file_name'])
                self.img_anns.append(anns)
                self.is_hor.append(3)

    def __getitem__(self, idx):
        # read img, and set yolo format label
        img_path = os.path.join(self.img_files[idx])
        img = Image.open(img_path).convert('RGB')

        if self.augment:
            img = random_enhance(img)  # Brightness,Color,Contrast

        img_anns = self.img_anns[idx]

        ori_w, ori_h = img.size
        img = img.resize((self.img_size_w, self.img_size_h), Image.ANTIALIAS)
        img = ToTensor()(img)
        if len(img.shape) != 3:  # gray to rgb
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        img, pad = pad_to_square(img, 0)  # return padded img and pad
        _, padded_h, padded_w = img.shape

        boxes1 = []
        labels1 = []
        for ann in img_anns:
            boxes1.append(cvt_xywh_2pts(ann['bbox']))  # x1y1x2y2
            labels1.append(ann['category_id'] + 1)

        boxes1 = torch.as_tensor(boxes1, dtype=torch.float32)
        labels1 = torch.as_tensor(labels1, dtype=torch.int64)

        # x1y1x2y2, box normalize to padded img
        x1 = (boxes1[:, 0] / ori_w * self.img_size_w + pad[0]) / padded_w
        y1 = (boxes1[:, 1] / ori_h * self.img_size_h + pad[2]) / padded_h
        x2 = (boxes1[:, 2] / ori_w * self.img_size_w + pad[0]) / padded_w
        y2 = (boxes1[:, 3] / ori_h * self.img_size_h + pad[2]) / padded_h

        # to xyywh, center x,y
        boxes1[:, 0] = (x1 + x2) / 2
        boxes1[:, 1] = (y1 + y2) / 2
        boxes1[:, 2] = (x2 - x1)
        boxes1[:, 3] = (y2 - y1)

        # yolo train targets
        targets = torch.zeros((boxes1.shape[0], 7))
        targets[:, 2:6] = boxes1  # location
        targets[:, 1] = labels1  # class
        targets[:, -1] = 1.

        lam = 1
        mix_img = img * lam  # ori no mix img

        # mixup, 0.3 ratio
        if self.mixup and np.random.random() < 0.3:
            alpha = 2.0
            lam = np.random.beta(alpha, alpha)
            # lam = 0.5
            # random choose another img
            idx2 = np.random.choice(np.delete(np.arange(self.__len__()), idx))

            img_path2 = os.path.join(self.img_files[idx2])
            img2 = Image.open(img_path2).convert('RGB')

            if self.augment:
                img2 = random_enhance(img2)

            img_anns2 = self.img_anns[idx2]

            ori_w2, ori_h2 = img2.size
            img2 = img2.resize((self.img_size_w, self.img_size_h), Image.ANTIALIAS)
            img2 = ToTensor()(img2)
            if len(img2.shape) != 3:
                img2 = img2.unsqueeze(0)
                img2 = img2.expand((3, img2.shape[1:]))

            img2, pad = pad_to_square(img2, 0)
            _, padded_h, padded_w = img2.shape

            boxes2 = []
            labels2 = []
            for ann in img_anns2:
                boxes2.append(cvt_xywh_2pts(ann['bbox']))
                labels2.append(ann['category_id'] + 1)

            boxes2 = torch.as_tensor(boxes2, dtype=torch.float32)
            labels2 = torch.as_tensor(labels2, dtype=torch.int64)

            # xyxy
            x1 = (boxes2[:, 0] / ori_w2 * self.img_size_w + pad[0]) / padded_w
            y1 = (boxes2[:, 1] / ori_h2 * self.img_size_h + pad[2]) / padded_h
            x2 = (boxes2[:, 2] / ori_w2 * self.img_size_w + pad[0]) / padded_w
            y2 = (boxes2[:, 3] / ori_h2 * self.img_size_h + pad[2]) / padded_h

            # to xywh
            boxes2[:, 0] = (x1 + x2) / 2
            boxes2[:, 1] = (y1 + y2) / 2
            boxes2[:, 2] = (x2 - x1)
            boxes2[:, 3] = (y2 - y1)

            # concat bbox together
            b1_len, b2_len = boxes1.shape[0], boxes2.shape[0]
            targets = torch.zeros((b1_len + b2_len, 7))
            targets[:b1_len, 2:6] = boxes1
            targets[:b1_len, 1] = labels1
            targets[:b1_len, -1] = lam
            targets[b1_len:, 2:6] = boxes2
            targets[b1_len:, 1] = labels2
            targets[b1_len:, -1] = 1 - lam

            mix_img = img * lam + (1 - lam) * img2

        # augment
        if self.augment and not self.aug_double:
            a = np.random.random()
            if a < 1.0 / 4:
                mix_img, targets = T.horisontal_flip(mix_img, targets)
            elif a < 2.0 / 4:
                mix_img, targets = T.vertical_flip(mix_img, targets)
            elif a < 3.0 / 4:
                mix_img, targets = T.horisontal_flip(mix_img, targets)
                mix_img, targets = T.vertical_flip(mix_img, targets)
        elif self.augment:
            if self.is_hor[idx] == 1:
                mix_img, targets = T.horisontal_flip(mix_img, targets)
            elif self.is_hor[idx] == 2:
                mix_img, targets = T.vertical_flip(mix_img, targets)
            elif self.is_hor[idx] == 3:
                mix_img, targets = T.horisontal_flip(mix_img, targets)
                mix_img, targets = T.vertical_flip(mix_img, targets)

        return mix_img, targets

    def collate_fn(self, batch):
        """ collate img list to a batch """
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]

        for i, boxes in enumerate(targets):  # 7dim: idx,class,coords,confidence
            boxes[:, 0] = i  # img idx in a batch

        targets = torch.cat(targets, 0)

        self.img_size = self.img_size
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))  # random size, 32x, step=32

        # as targets are normalized, no need to chagne targets
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        self.batch_count += 1

        return imgs, targets

    def __len__(self):
        return len(self.img_files)
