import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from terminaltables import AsciiTable
import torch
from torchvision.transforms import functional as F
from PIL import Image

from asm.dataset.utils import change_size_for_yolo, rescale_boxes
from asm.net import yolov3, faster_rcnn
from asm.net.utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
from pprint import pprint

# params
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# label_names = ['条烟', '包烟']  # matlab miss fonts
label_names = ['bg', 'cigar_A', 'cigar_a', ]
label_colors = [[0, 0, 0], [246, 126, 15], [219, 0, 28]]

cnt = 0


def evaluate_yolo(model, dataloader, iou_thres, conf_thres, nms_thres, batch_size, img_size):
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract la
        # bels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        imgs = imgs.type(Tensor)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0], dtype=np.int)
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def get_yolov3_map(model, dataloader, epoch, logger, debug=False):
    precision, recall, AP, f1, ap_class = evaluate_yolo(
        model,
        dataloader,
        iou_thres=0.5,
        conf_thres=0.7,
        nms_thres=0.5,
        img_size=416,
        batch_size=1,
    )
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]
    if epoch is not None:
        logger.list_of_scalars_summary(evaluation_metrics, epoch)
    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, cat_names[c], "%.5f" % AP[i]]]
    if debug:
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")  # f, format shorcut
        print(f"---- reall{recall.mean()}")
        print(f"---- precision{precision.mean()}")
    return AP.mean(), recall.mean()


def parse_rcnn_detection(detection, score_thre=0.7, vis=True, save_dir=None):
    boxes = detection['boxes'].cpu().numpy()
    labels = detection['labels'].cpu().numpy()  # label idxs
    scores = detection['scores'].cpu().numpy()
    keep_idxs = np.where(scores > score_thre)[0]

    result = {
        'labels': labels[keep_idxs],  # map color
        'boxes': boxes[keep_idxs]
    }
    pprint(result)
    global cnt
    if vis:
        plt.figure(figsize=(10, 6))
        plt.imshow(img)

        for idx in range(len(result['boxes'])):
            label, box = result['labels'][idx], result['boxes'][idx]  # label idx
            # box
            plt.gca().add_patch(plt.Rectangle(xy=(box[0], box[1]),
                                              width=box[2] - box[0],
                                              height=box[3] - box[1],
                                              edgecolor=[c / 255 for c in label_colors[label]],
                                              fill=False, linewidth=2))
            # name
            plt.annotate(label_names[label],
                         xy=(box[0], box[1]), fontsize=10,
                         xycoords='data', xytext=(2, 5), textcoords='offset points',
                         bbox=dict(boxstyle='round, pad=0.3',  # linewidth=0 可以不显示边框
                                   facecolor=[c / 255 for c in label_colors[label]], lw=0),
                         color='w')
        if save_dir and len(result['boxes']) > 0:
            cnt += 1
            plt.savefig(os.path.join(save_dir, '{}.png'.format(cnt)),
                        bbox_inches='tight')

        # plt.show()  # out for, show img if no bboxes detected


def test_faster_rcnn(device, img, backbone, epoch, num_class):
    ckpt = 'output/{}/{}_epoch_{}.pth'.format(backbone, backbone, epoch)
    print('load', ckpt)

    model = faster_rcnn.get_model(backbone, num_class, self_pretrained=True)
    model.load_state_dict(torch.load(ckpt))
    model.to(device)
    print('done!')

    # save results
    vis_dir = 'vis/{}/epoch_{}'.format(backbone, epoch)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    with torch.no_grad():
        model.eval()

        img_tensor = F.to_tensor(img)
        detection = model([img_tensor.to(device)])[0]  # [ ] equals unsqueeze(0)
        parse_rcnn_detection(detection, save_dir=vis_dir)


def test_yolov3_tiny(device, img, score_thre, nms_thres):
    config_path = './config/yolov3-tiny-cigar2.cfg'
    backbone = 'yolov3-tiny'
    epoch = 161
    vis = True

    model = yolov3.Darknet(config_path, 416)
    ckpt = os.path.join(os.path.dirname(__file__), 'output/{}/{}-{}.pth'.format(backbone, backbone, epoch))

    if device is not 'cpu':
        model.load_state_dict(torch.load(ckpt))

    else:
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))

    model.to(device)
    print('done!')
    model.eval()

    vis_dir = 'vis/{}/epoch_{}'.format(backbone, epoch)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    global cnt

    with torch.no_grad():
        model.eval()
        # img = np.array(img)
        w, h = img.size

        img_tensor = change_size_for_yolo(img, size_w=416, size_h=416).unsqueeze(0)

        detections = model(img_tensor.to(device))

        detections = non_max_suppression(detections, score_thre, nms_thres)[0]

        boxes = None
        labels = None
        if detections is not None:
            detections = rescale_boxes(detections, 416, (h, w))
            boxes = detections[:, :4].cpu().numpy()  # x1,y1,x2,y2
            labels = detections[:, -1].cpu().numpy()  # +1

        if vis and boxes is not None:
            plt.figure(figsize=(10, 6))
            plt.imshow(img)

            for idx in range(len(boxes)):
                label, box = int(labels[idx]), boxes[idx]  # label idx
                # box
                plt.gca().add_patch(plt.Rectangle(xy=(box[0], box[1]),
                                                  width=box[2] - box[0],
                                                  height=box[3] - box[1],
                                                  edgecolor=[c / 255 for c in label_colors[label]],
                                                  fill=False, linewidth=2))
                # name
                plt.annotate(label_names[label],
                             xy=(box[0], box[1]), fontsize=10,
                             xycoords='data', xytext=(2, 5), textcoords='offset points',
                             bbox=dict(boxstyle='round, pad=0.3',  # linewidth=0 可以不显示边框
                                       facecolor=[c / 255 for c in label_colors[label]], lw=0),
                             color='w')
            if vis_dir and len(boxes) > 0:
                cnt += 1
                plt.savefig(os.path.join(vis_dir, '{}.png'.format(cnt)),
                            bbox_inches='tight')

            # plt.show()  # out for, show img if no bboxes detected
        # return boxes, labels


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')

    else:
        device = torch.device('cpu')

    imgdir = './data/RetailProductDataset/images/'
    img_paths = os.listdir(imgdir)
    num = 50
    sample_ids = np.sort(np.random.randint(0, len(img_paths), size=num))

    for img_idx in sample_ids:
        img = Image.open(os.path.join(imgdir, img_paths[img_idx])).convert('RGB')
        # test_faster_rcnn(device, img, backbone='res50', epoch=20, num_class=3)

        test_yolov3_tiny(device, img, score_thre=0.7, nms_thres=0.7)
