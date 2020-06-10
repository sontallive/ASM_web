import os
import torch
from asm.datasets.VOC import VOC_Dataset
from asm.datasets.transforms import get_transform
from asm.net.faster_rcnn import get_model
from asm.tools.utils import collate_fn
from asm.tools.engine import train_one_epoch, evaluate
from tensorboardX import SummaryWriter

gpus = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

if __name__ == '__main__':
    # 1.dataset
    dataset = VOC_Dataset(data='VOC2007', split='trainval',
                          transforms=get_transform(True))
    dataset_eval = VOC_Dataset(data='VOC2007', split='test',
                               transforms=get_transform(False))  # to tensor
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=collate_fn)
    # 2.model
    backbone = 'mobile'
    num_classes = 21  # bg + 20
    model = get_model(backbone, num_classes, self_pretrained=False)
    # whether use data parallel
    if len(gpus) > 1:  # !!!
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()  # maybe this code not support data parallel?

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]  # conv1,conv2_x not update
    optimizer = torch.optim.SGD(params,
                                lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # lr scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)  # 1/10
    # model save and tensor board
    model_save_dir = os.path.join('output', backbone)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    writer = SummaryWriter()

    num_epochs = 20
    check_step = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, epoch,
                        print_freq=2,
                        writer=writer, begin_step=epoch * len(data_loader))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_eval)
        if (epoch + 1) % check_step == 0:
            torch.save(model.state_dict(), os.path.join(model_save_dir, '{}_epoch_{}.pth'.format(backbone, epoch + 1)))
