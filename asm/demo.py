import torch
from torchvision.transforms import functional as F
from asm.net import faster_rcnn, yolov3
from asm.net.utils import non_max_suppression
from asm.dataset.utils import change_size_for_yolo, rescale_boxes
import numpy as np
import os
import cv2

# /nfs/xs/Codes/VIPA_ASM/models
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
# device can be global, but model can't be,
# cus we may use multi models or restart a train with latest model, need a return
device = None


def load_detection_model(dataset, model_name, num_classes, ckpt_epoch='latest'):
    """
    :param dataset: Cigar, VOC
    :param model_name: faster_rcnn_res50, faster_rcnn_mobile, yolov3_tiny
    :param num_classes:
    :param ckpt_epoch: default load the latest model
    """
    global device
    # model
    print('load detection model...')
    ckpt_path = os.path.join(model_dir, dataset, model_name, '{}_{}.pth'.format(model_name, ckpt_epoch))

    if 'faster_rcnn' in model_name:
        model = faster_rcnn.get_model(model_name, num_classes, self_pretrained=True)
    elif model_name == 'yolov3-tiny':
        config = os.path.join(os.path.dirname(__file__), 'config/{}-cigar2.cfg'.format(model_name))
        # todo: change config yolo layer class
        model = yolov3.Darknet(config, 416)
    else:
        raise ValueError('not implement!')

    # ckpt
    print('load {}'.format(ckpt_path))
    # load to gpu/cpu
    if torch.cuda.is_available():
        # print('use gpu')
        device = torch.device('cuda')
        model.load_state_dict(torch.load(ckpt_path))
    else:
        # print('use cpu')
        device = torch.device('cpu')
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    model.to(device)
    print('load done!')

    return model


@torch.no_grad()
def faster_rcnn_detect(model, img, score_thre=0.7):
    model.eval()
    # img: ``PIL Image`` or ``numpy.ndarray``, cv bgr will convert to rgb
    img_tensor = F.to_tensor(img)
    detection = model([img_tensor.to(device)])[0]  # only 1 img

    # parse detection
    boxes = detection['boxes'].cpu().numpy()  # x1,y1,x2,y2
    labels = detection['labels'].cpu().numpy()  # label idxs
    scores = detection['scores'].cpu().numpy()
    keep_idxs = np.where(scores > score_thre)[0]

    return boxes[keep_idxs], labels[keep_idxs]


@torch.no_grad()
def yolov3_tiny_detect(model, img, score_thre=0.5, nms_thres=0.5):
    model.eval()
    ori_w, ori_h = img.size
    img_tensor = change_size_for_yolo(img, size_w=416, size_h=416).unsqueeze(0)  # chang size and to tensor
    detections = model(img_tensor.to(device))

    detections = non_max_suppression(detections, score_thre, nms_thres)[0]

    boxes, labels = np.array([]), np.array([])
    if detections is not None:
        detections = rescale_boxes(detections, 416, (ori_h, ori_w))
        boxes = detections[:, :4].cpu().numpy()  # x1,y1,x2,y2
        labels = detections[:, -1].cpu().numpy()  # +1

    return boxes, labels


def cal_fps(func, param, s=10):
    # print('{}:'.format(cap_fuc.__name__))  # 打印函数名称
    cnt_time = cnt_frames = 0
    tick_frequency = cv2.getTickFrequency()  # 1000000000.0
    while True:
        t1 = cv2.getTickCount()
        func(param.to(device))
        t2 = cv2.getTickCount()

        cnt_frames += 1
        cnt_time += (t2 - t1) / tick_frequency
        print('\r{}/{}'.format(cnt_frames, cnt_time), end='')

        if cnt_time >= s:
            fps = cnt_frames / s
            print('\navg fps:', fps)
            return fps
