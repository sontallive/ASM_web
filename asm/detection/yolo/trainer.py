from asm.detection.trainer import trainer
from asm.detection.yolo.cats import cat_names
from asm.dataset.detection_dataset import Detection_Dataset_yolo
from asm.tools import utils
from asm.net import yolov3
import datetime
import os
import time
import torch
from torch.autograd import Variable
from terminaltables import AsciiTable
from asm.test import evaluate_yolo
from shutil import rmtree


class yolo_trainer(trainer):

    def __init__(self, name='yolo'):
        super(yolo_trainer, self).__init__()
        self.tricks = ['mixup', 'multiscale', 'augment', 'cosine', 'aug_double']
        self.name = name

    def warmup_lr_scheduler(self, optimizer, warmup_iters, warmup_factor):

        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    def get_yolov3_map(self,model, dataloader, epoch, logger, debug=False):
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
            print(f"---- mAP {AP.mean()}")
            print(f"---- reall{recall.mean()}")
            print(f"---- precision{precision.mean()}")
        return AP.mean(), recall.mean()

    def load_dataset(self, root, data, batch_size=2, augment=False, multiscale=False, mixup=False,
                                 aug_double=False):
        dataset_train = Detection_Dataset_yolo(root, data,
                                               img_size_w=416,
                                               img_size_h=416,
                                               augment=augment,
                                               multiscale=multiscale,
                                               normalized_labels=False,
                                               mixup=mixup,
                                               aug_double=aug_double)

        dataloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset_train.collate_fn,
        )
        return dataloader

    def load_pretrained_parameter(self, path):
        pass

    def train(self, root_path, data_root, train_data, valid_data, test_data,
              gpuid, epochs, tricks=['mixup', 'multiscale', 'augment', 'cosine', 'aug_double'],
              gradient_accumulations=1, checkpoint_interval=5, evaluation_interval=1, debug=False,
              pretrained_path=None, start_epoch=0, dataset_name=None,
              update_fn=None):
        best_val_map = 0
        self.clear_training_status()
        self.update_fn = update_fn
        if dataset_name is None:
            output_dir = os.path.join(root_path, 'output/yolov3-tiny')
        else:
            output_dir = os.path.join(root_path, 'output', dataset_name, 'yolov3-tiny')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # update status before start ##############
        status = {"status":"training",
                  "epoch_total":epochs}
        if self.update_status(status):
            return
        ###################################################

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

        self.training_status['tb_path'] = os.path.join(root_path, 'logs/yolov3-tiny')
        if not os.path.exists(self.training_status['tb_path']):
            os.makedirs(self.training_status['tb_path'])
        try:
            for file in os.listdir(self.training_status['tb_path']):
                file_path = os.path.join(self.training_status['tb_path'],file)
                os.remove(file_path)
        except Exception:
            pass
        logger = utils.Logger(self.training_status['tb_path'])

        print(self.training_status['tb_path'])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_def = os.path.join(root_path, 'asm/config/yolov3-tiny-cigar2.cfg')

        # update status before LOAD MODEL ##############
        status = {"stage": "Loading Model"}
        if self.update_status(status):
            return
        ###################################################
        model = yolov3.Darknet(model_def).to(device)
        model.apply(utils.weights_init_normal)
        self.model = model
        # update status before Loading Pretrained Weight ##############
        status = {"stage": "Loading Pretrained Weight"}
        if self.update_status(status):
            return
        ###################################################
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            model.load_state_dict(checkpoint)
        else:
            pretrained_weights = os.path.join(root_path, 'asm/weights/yolov3-tiny.weights')
            model.load_darknet_weights(pretrained_weights)

        # update status before LOAD DATASET ##############
        status = {"stage": "Loading Dataset"}
        if self.update_status(status):
            return
        ###################################################
        data_train_loader = self.load_dataset(data_root, train_data, batch_size=2,
                                                          augment=('augment' in tricks),
                                                          multiscale='multiscale' in tricks,
                                                          mixup='mixup' in tricks,
                                                          aug_double='aug_double' in tricks)
        data_valid_loader = self.load_dataset(data_root, valid_data, batch_size=1)
        data_test_loader = self.load_dataset(data_root, test_data, batch_size=1)

        optimizer = torch.optim.Adam([{'params':model.parameters(),'initial_lr':float(model.hyperparams['learning_rate'])}],
                                     lr=float(model.hyperparams['learning_rate']),
                                     weight_decay=float(model.hyperparams['decay']))

        if 'cosine' in tricks:
            lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                                     last_epoch=start_epoch)

        metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]

        warmup_factor = 1. / 1000
        warmup_iters = min(1000, 10 - 1)
        lr_schedule2 = self.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        for epoch in range(epochs):
            model.train()
            start_time = time.time()
            total_loss = 0

            # update training status before epoch start    ##
            status = {
                        "epoch_current": epoch + 1,
                        "epoch_trained": epoch,
                        "stage": "training"
                     }
            if self.update_status(status):
                return
            ##################################################
            for batch_i, (imgs, targets) in enumerate(data_train_loader):

                # update training status before epoch start    ##
                status = {"stage": 'training on batch_{}/{}'.format(batch_i, len(data_train_loader))}
                if self.update_status(status):
                    return
                ##################################################


                batches_done = len(data_train_loader) * epoch + batch_i
                # print(imgs.shape)
                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)

                loss, outputs = model(imgs, targets)
                loss.backward()

                if batches_done % gradient_accumulations == 0:
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

                    # print("batches done:{}".format(batches_done))
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"
                total_loss += loss.item()

                # update training status after train loss        ##
                self.training_status['msgs']['train loss'] = total_loss
                status = {"stage": 'evaluating on train'}
                if self.update_status(status):
                    return
                ##################################################

                # Determine approximate time left for epoch
                epoch_batches_left = len(data_train_loader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"

                # print(log_str)

                model.seen += imgs.size(0)
            if debug:
                print("\n---- [Epoch %d/%d] ----\n" % (epoch, epochs))
                print("total loss in train dataset:", total_loss)
            self.training_status['loss'] = total_loss

            if epoch % evaluation_interval == 0:
                train_map, train_recall = self.get_yolov3_map(model,
                                                              data_train_loader,
                                                              epoch=None,
                                                              logger=None,
                                                              debug=debug)

                # update training status after evaluate              ##
                self.training_status['stage'] = 'evaluating on valid'
                self.training_status['msgs']['train map'] = train_map
                self.training_status['msgs']['train recall'] = train_recall
                if update_fn:
                    update_fn(self.training_status)
                    if self.need_stop:
                        return
                #######################################################
                if debug:
                    print("---- Evaluating Model ----")
                # Evaluate the model on the validation set
                valid_map, valid_recall = self.get_yolov3_map(model,
                                                              data_valid_loader,
                                                              epoch,
                                                              logger,
                                                              debug=debug)
                if valid_map > best_val_map:
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, "best.pth"))
                # update training status after evaluate         ##
                self.training_status['msgs']['valid map'] = valid_map
                self.training_status['msgs']['valid recall'] = valid_recall
                if self.update_status({}):
                    return
                ##################################################

            output_dir = os.path.join(root_path, 'output/yolov3-tiny')
            if epoch % checkpoint_interval == 0:
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                torch.save(model.state_dict(),
                           os.path.join(output_dir, f"%d_test.pth" % epoch))

        torch.save(model.state_dict(),
                   os.path.join(output_dir, "latest.pth"))

        # update training status before evaluate  test       ##
        self.training_status['epoch_trained'] = epochs
        self.training_status['stage'] = 'evaluating on test'
        if update_fn:
            update_fn(self.training_status)
        ##################################################
        test_map, test_recall = self.get_yolov3_map(model,
                                                    data_test_loader,
                                                    epoch=None,
                                                    logger=None,
                                                    debug=debug)
        # update training status after evaluate  test       ##
        self.training_status['stage'] = 'train done'
        self.training_status['status'] = 'idle'
        self.training_status['msgs']['test map'] = test_map
        self.training_status['msgs']['test recall'] = test_recall
        if update_fn:
            update_fn(self.training_status)
        ##################################################

        return self.training_status


if __name__ == '__main__':
    train_data = 'Cigar_train_super.json'
    valid_data = train_data.replace('train', 'val')
    test_data = train_data.replace('train', 'test')
    trainer = yolo_trainer()

    root_path = os.path.join(os.getcwd(), '../../../')
    dataset_dir = os.path.join(root_path, 'datasets/Cigar')
    trainer.train(root_path,
                  dataset_dir,
                  train_data,
                  valid_data,
                  test_data,
                  gpuid='1',
                  epochs=5,
                  tricks=['mixup', 'augment'],
                  debug=True)