import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator


def faster_rcnn_mobile(self_pretrained, num_classes):
    if not self_pretrained:
        print('load mobilenet_v2 backbone pretrained on ImageNet')

    pretrained = False if self_pretrained else True  # pretrain on ImageNet
    backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
    backbone.out_channels = 1280

    # 考虑更多长条形
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))  # todo: add 0.3 4.0

    # need this, as mobilenet out 1 level features
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],  # can be multi
                                                    output_size=7,
                                                    sampling_ratio=2)
    # model will do normalize and resize itself
    # box_nms_thresh used during inference
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def faster_rcnn_res50(self_pretrained, num_classes):
    if not self_pretrained:
        print('load res50 backbone pretrained on COCO')

    pretrained = False if self_pretrained else True  # pretrain on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features,
                                                      num_classes=num_classes)
    return model


def get_model(backbone, num_classes, self_pretrained=False):
    if backbone == 'faster_rcnn_res50':
        return faster_rcnn_res50(self_pretrained, num_classes)
    elif backbone == 'faster_rcnn_mobile':
        return faster_rcnn_mobile(self_pretrained, num_classes)
    else:
        raise ValueError('no such backbone!')
