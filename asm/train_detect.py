import os
import time
import datetime
import torch
from tensorboardX import SummaryWriter
from asm.tools.engine import train_one_epoch, evaluate
from asm.dataset.detection_dataset import Detection_Dataset_rcnn, Detection_Dataset_yolo
from asm.dataset import transforms as T
from asm.net import yolov3, faster_rcnn
from asm.tools import utils
from asm.test import get_yolov3_map
from terminaltables import AsciiTable
from torch.autograd import Variable

model_tricks = {
    "faster_rcnn": ['cosine', 'augment'],
    "yolov3-tiny": ['mixup', 'multiscale', 'augment', 'cosine', 'aug_double']
}

model_list = list(model_tricks.keys())

training_status = {
    'epoch_current': 0,
    'epoch_total': 10,
    'tb_path': '',
    'loss': 0,
    'train_map': 0,
    'valid_map': 0,
    'test_map': 0,
    'valid_recall': 0,
    'train_recall': 0,
    'test_recall': 0
}


def clear_training_statue():
    training_status['loss'] = 0
    training_status['train_map'] = 0
    training_status['valid_map'] = 0
    training_status['test_map'] = 0
    training_status['train_recall'] = 0
    training_status['valid_recall'] = 0
    training_status['test_recall'] = 0


def get_status():
    return training_status


def stop():
    training_status['status'] = 'stop'


def get_transform(train, aug_double):
    transforms = []
    if train:
        transforms.append(T.RandomEnhance())  # enhance Brightness,Color,Contrast
    transforms.append(T.ToTensor())  # only img to_tensor
    if train:
        if not aug_double:  # cus aug_double has done flip
            transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_faster_rcnn_dataset(root, data, batch_size=2,
                             augment=False, aug_double=False):
    # train default use augment
    dataset = Detection_Dataset_rcnn(root, data,
                                     transforms=get_transform(train=augment, aug_double=aug_double))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4,  # utils collate_fn, cus target is not simple
                                              collate_fn=utils.collate_fn)
    return data_loader


def load_yolov3_tiny_dataset(root, data, batch_size=2,
                             augment=False, multiscale=False, mixup=False, aug_double=False):
    dataset = Detection_Dataset_yolo(root, data,
                                     img_size_w=416,
                                     img_size_h=416,
                                     augment=augment,  # actually, transform has done in __getitem__
                                     multiscale=multiscale,
                                     normalized_labels=False,
                                     mixup=mixup,
                                     aug_double=aug_double)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    return dataloader


def train_yolov3_tiny(root,
                      train_data, valid_data, test_data,
                      epochs,
                      tricks=None,  # todo: change class in cfg
                      gradient_accumulations=1,
                      checkpoint_interval=5,
                      evaluation_interval=1,
                      debug=False):
    clear_training_statue()
    if tricks is None:
        tricks = model_tricks['yolov3-tiny']

    training_status['status'] = 'training'
    training_status['epoch_total'] = epochs

    logger = utils.Logger("logs/yolov3-tiny")
    training_status['tb_path'] = os.getcwd() + '/logs/yolov3-tiny'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_def = './config/yolov3-tiny-cigar2.cfg'
    pretrained_weights = './weights/yolov3-tiny.weights'

    # build model
    model = yolov3.Darknet(model_def).to(device)
    model.apply(utils.weights_init_normal)
    model.load_darknet_weights(pretrained_weights)

    # load data
    data_train_loader = load_yolov3_tiny_dataset(root, train_data, batch_size=2,
                                                 augment='augment' in tricks,
                                                 multiscale='multiscale' in tricks,
                                                 mixup='mixup' in tricks,
                                                 aug_double='aug_double' in tricks)
    data_valid_loader = load_yolov3_tiny_dataset(root, valid_data, batch_size=1)
    data_test_loader = load_yolov3_tiny_dataset(root, test_data, batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(model.hyperparams['learning_rate']),
                                 weight_decay=float(model.hyperparams['decay']))

    if 'cosine' in tricks:
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    metrics = [
        "grid_size",
        "loss",
        "x", "y", "w", "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # normal lr schedule
    warmup_factor = 1. / 1000
    warmup_iters = min(1000, 10 - 1)
    lr_schedule2 = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_loss = 0
        training_status['epoch_current'] = epoch + 1
        for batch_i, (imgs, targets) in enumerate(data_train_loader):
            if training_status['status'] == 'stop':
                return training_status
            batches_done = len(data_train_loader) * epoch + batch_i
            imgs = imgs.to(device)
            targets = targets.to(device)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
                if epoch < 10:
                    lr_schedule2.step()
                elif 'cosine' in tricks:
                    lr_schedule.step()

            # ----------------
            #   Log progress
            # ----------------
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(data_train_loader))
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            total_loss += loss.item()
            # Determine approximate time left for epoch
            epoch_batches_left = len(data_train_loader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            # print(log_str)

            model.seen += imgs.size(0)
        if debug:
            print("\n---- [Epoch %d/%d] ----\n" % (epoch, epochs))
            print("total loss in train dataset:", total_loss)
        training_status['loss'] = total_loss

        if epoch % evaluation_interval == 0:
            training_status['train_map'], training_status['train_recall'] = get_yolov3_map(model, data_train_loader, epoch=None, logger=None, debug=debug)
            if debug:
                print("---- Evaluating Model ----")
            # Evaluate the model on the validation set
            training_status['valid_map'], training_status['valid_recall'] = get_yolov3_map(model, data_valid_loader, epoch, logger, debug=debug)

        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(),
                       f"output/yolov3-tiny/%d_test.pth" % epoch)

    training_status['test_map'], training_status['test_recall'] = get_yolov3_map(model, data_test_loader, epoch=None, logger=None, debug=debug)
    training_status['status'] = 'stop'

    return training_status


def train_faster_rcnn(root,
                      train_data, valid_data, test_data,
                      epochs,
                      tricks=None,
                      backbone='faster_rcnn_res50', num_classes=3, check_step=5, debug=False):
    clear_training_statue()
    if tricks is None or tricks[0] == '':
        tricks = model_tricks['faster_rcnn']

    training_status['status'] = 'training'
    training_status['epoch_total'] = epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load data
    data_train_loader = load_faster_rcnn_dataset(root, train_data, batch_size=2,
                                                 augment='augment' in tricks,
                                                 aug_double='aug_double' in tricks)
    data_valid_loader = load_faster_rcnn_dataset(root, valid_data, batch_size=1)
    data_test_loader = load_faster_rcnn_dataset(root, test_data, batch_size=1)

    model = faster_rcnn.get_model(backbone, num_classes, self_pretrained=False)
    model.to(device)

    model_save_dir = os.path.join('output', backbone)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    params = [p for p in model.parameters() if p.requires_grad]  # if not update backbone
    optimizer = torch.optim.SGD(params,
                                lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)

    if 'cosine' in tricks:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=epochs)
    else:
        # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)  # 1/10

    writer = SummaryWriter('logs/faster_rcnn')
    training_status['tb_path'] = os.getcwd() + '/logs/faster_rcnn'

    for epoch in range(epochs):
        training_status['epoch_current'] = epoch + 1
        if training_status['status'] == 'stop':
            return training_status
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_train_loader, device, epoch,
                        print_freq=10,
                        writer=writer, begin_step=epoch * len(data_train_loader))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_valid_loader, device=device)
        if (epoch + 1) % check_step == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, '{}_epoch_{}.pth'.format(backbone, epoch + 1)))

    evaluate(model, data_test_loader, device=device)
    training_status['status'] = 'stop'
    return training_status


if __name__ == '__main__':
    train_data = 'train_coco.json'
    valid_data = 'val_coco.json'
    test_data = 'test_coco.json'

    train_faster_rcnn('./',
                      train_data,
                      valid_data,
                      test_data,
                      gpuid='1',
                      num_epochs=1,
                      tricks=['cosine', 'augment'],
                      backbone='res50',  # 'mobile'
                      num_classes=3,
                      check_step=5)

    print(training_status)

    # train_yolov3_tiny('./',
    #                     train_data,
    #                   valid_data,
    #                   test_data,
    #                   gpuid='1',
    #                   epochs=1,
    #                   tricks=['mixup', 'multiscale', 'aug_double', 'augment', 'cosine'])
    #
    # print(training_status)
