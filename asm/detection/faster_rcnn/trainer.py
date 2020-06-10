from asm.detection.trainer import trainer
import torch
from asm.dataset.detection_dataset import Detection_Dataset_rcnn
from asm.tools import utils
from asm.net.faster_rcnn import get_model
import os
from tensorboardX import SummaryWriter
from asm.tools.engine import train_one_epoch, evaluate
from asm.dataset import transforms as T
from shutil import rmtree


class fast_rcnn_trainer(trainer):

    def __init__(self, name='fast_rcnn'):
        super(fast_rcnn_trainer, self).__init__()
        self.tricks = ['cosine', 'augment']
        self.name = name

    def get_transform(self, train, aug_double):
        transforms = []
        if train:
            transforms.append(T.RandomEnhance())
        transforms.append(T.ToTensor())  # only img to_tensor
        if train:
            if not aug_double:
                transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def load_dataset(self, root_path, data, batch_size=2, aug=False, aug_double=False):
        dataset = Detection_Dataset_rcnn(root_path,
                                         data,
                                         transforms=self.get_transform(train=aug, aug_double=aug_double)
                                         )

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4,
                                                  collate_fn=utils.collate_fn)
        return data_loader

    def train(self, root_path, data_root, train_data, valid_data, test_data,
              gpuid, num_epochs, tricks=['cosine', 'augment'],
              pretrained_path=None, start_epoch=0, dataset_name=None,
              backbone='faster_rcnn_res50', num_classes=3, check_step=5, update_fn=None):
        self.clear_training_status()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

        self.training_status['status'] = 'training'
        self.training_status['epoch_total'] = num_epochs
        if dataset_name is None:
            model_save_dir = os.path.join(root_path, 'output', backbone)
        else:
            model_save_dir = os.path.join(root_path, 'output', dataset_name, backbone)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # update training status before loading dataset ##
        self.training_status['stage'] = 'Loading dataset'
        if update_fn:
            update_fn(self.training_status)
            if self.need_stop:
                return
        ##################################################
        data_train_loader = self.load_dataset(data_root, train_data,
                                              batch_size=2, aug='augment' in tricks,
                                              aug_double='aug_double' in tricks)
        data_valid_loader = self.load_dataset(data_root, valid_data, batch_size=1)
        data_test_loader = self.load_dataset(data_root, test_data, batch_size=1)

        # update training status before loading model ##
        self.training_status['stage'] = 'Loading model'
        if update_fn:
            update_fn(self.training_status)
            if self.need_stop:
                return
        #################################################
        model = get_model(backbone, num_classes, self_pretrained=False)
        model.to(device)
        self.model = model
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            model.load_state_dict(checkpoint)

        model_save_dir = os.path.join(root_path, 'output', backbone)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        params = [p for p in model.parameters() if p.requires_grad]  # if not update backbone
        optimizer = torch.optim.SGD([{'params': params, 'initial_lr': 0.005}],
                                    lr=0.005,
                                    momentum=0.9,
                                    weight_decay=0.0005)

        if 'cosine' in tricks:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,
                                                                      last_epoch=start_epoch)
        else:
            # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=3,
                                                           gamma=0.1,
                                                           last_epoch=start_epoch)  # 1/10

        self.training_status['tb_path'] = os.path.join(root_path, 'logs/faster_rcnn')

        if not os.path.exists(self.training_status['tb_path']):
            os.makedirs(self.training_status['tb_path'])
        try:
            for file in os.listdir(self.training_status['tb_path']):
                file_path = os.path.join(self.training_status['tb_path'], file)
                os.remove(file_path)
        except Exception:
            pass
        writer = SummaryWriter(self.training_status['tb_path'])

        print('tb_path:', self.training_status['tb_path'])

        for epoch in range(num_epochs):
            # update training status before epoch start    ##
            status = {
                'epoch_current': epoch + 1,
                'epoch_trained': epoch,
                'stage': 'training'
            }
            if self.update_status(status):
                return
            ##################################################

            # train for one epoch, printing every 10 iterations
            self.training_status['msgs']['loss'] = train_one_epoch(model, optimizer, data_train_loader, device, epoch,
                                                                  print_freq=10,
                                                                  writer=writer,
                                                                  begin_step=epoch * len(data_train_loader),
                                                                  )

            # update the learning rate
            lr_scheduler.step()

            # update training status before evaluate train  ##
            self.training_status['stage'] = 'Evaluate on train'
            if self.update_status():
                return
            ##################################################
            train_eval_msg = evaluate(model, data_train_loader, writer=writer)

            # update status before evaluate validation ##############
            self.training_status['msgs'].update({"train:" + k: v for k, v in train_eval_msg.items()})
            self.training_status['stage'] = 'evaluate on validation'
            if self.update_status():
                return
            ###################################################
            # evaluate on the test dataset
            val_eval_msg = evaluate(model, data_valid_loader, writer=writer)
            self.training_status['msgs'].update({"val:" + k: v for k, v in val_eval_msg.items()})

            if (epoch + 1) % check_step == 0:
                # update status before save model              ####
                self.training_status['stage'] = 'Saving Model...'
                if self.update_status():
                    return
                ###################################################
                torch.save(model.state_dict(),
                           os.path.join(model_save_dir, '{}_epoch_{}.pth'.format(backbone, epoch + 1)))

        torch.save(model.state_dict(),
                   os.path.join(model_save_dir, 'latest.pth'))
        # update training status before final test eval  ##
        self.training_status['epoch_trained'] = num_epochs
        self.training_status['stage'] = 'Final evaluate on test'
        self.update_status()
        ##################################################
        test_eval_msg = evaluate(model, data_test_loader, writer=writer)
        self.training_status['msgs'].update({"test:" + k: v for k, v in test_eval_msg.items()})
        self.training_status['status'] = 'idle'
        writer.close()

        self.update_status()
        return self.training_status


if __name__ == '__main__':
    # train_data = 'train_coco.json'
    train_data = 'Cigar_train_super.json'
    valid_data = train_data.replace('train', 'val')
    test_data = train_data.replace('train', 'test')
    trainer = fast_rcnn_trainer()

    root_path = os.path.join(os.getcwd(), '../../../')
    dataset_dir = os.path.join(root_path, 'datasets/Cigar')
    trainer.train(root_path,
                  dataset_dir,
                  train_data,
                  valid_data,
                  test_data,
                  gpuid='1',
                  num_epochs=5,
                  tricks=['cosine', 'augment'],
                  backbone='faster_rcnn_res50',  # 'res50'
                  pretrained_path=" ",
                  num_classes=3,
                  check_step=5)