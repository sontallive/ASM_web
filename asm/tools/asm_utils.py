from torchvision.transforms import functional as F
import torch
import cv2
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def infer(model, img_path, score_thre=0.7):
    img = cv2.imread(img_path)
    img_tensor = F.to_tensor(img)
    detection = model([img_tensor.cuda()])[0]  # only 1 img

    # parse detection
    boxes = detection['boxes'].cpu().numpy().astype(int)
    labels = detection['labels'].cpu().numpy().astype(int)  # label idxs
    scores = detection['scores'].cpu().numpy()

    keep_idxs = np.where(scores > score_thre)[0]

    return boxes[keep_idxs], labels[keep_idxs]


@torch.no_grad()
def detect_unlabel_imgs(model, detect_idxs, anns, CONF_THRESH=0.7):
    """
    :param model: on-training model
    :param detect_idxs: unlabel_idxs in anns
    :param anns: voc2012 trainval ann list
    :param CONF_THRESH: todo: may need increase as model performance increase?
    """
    model.eval()
    al_idxs, sl_idxs = [], []
    al_anns, sl_anns = [], []
    for idx in tqdm(detect_idxs):
        ann = anns[idx]
        boxes, labels = infer(model, ann['filepath'], score_thre=CONF_THRESH)
        # todo: this judgement is img-level, not box-class-level, so may not work well
        if len(labels) > 0:  # exists confident boxes
            # SL replace human label with model label
            sl_idxs.append(idx)
            ann['boxes'] = boxes
            ann['labels'] = labels - 1  # as detect_idx has +1
            sl_anns.append(ann)
        else:
            # AL still use human label
            al_idxs.append(idx)
            sl_anns.append(ann)

    return sl_idxs, al_idxs, sl_anns, al_anns
