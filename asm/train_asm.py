import os
import torch
from asm.datasets.VOC import VOC_Dataset
from asm.datasets.transforms import get_transform
from asm.datasets.voc_parser import load_data, dump_data  # load or dump updated voc2012 anns
from asm.net.faster_rcnn import get_model
from asm.tools.utils import collate_fn
from asm.tools.engine import train_one_epoch, evaluate
from asm.tools.asm_utils import detect_unlabel_imgs
from tensorboardX import SummaryWriter  # tensorboardX
import argparse


def parse_args(params):
    parser = argparse.ArgumentParser(description='Simple ASM with FasterRCNN on VOC2012')
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--backbone', default='res50', type=str, help='res50, mobile')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--pretrain_epoch', default=10, type=int, help='pretrained model on VOC2007')
    parser.add_argument('--num_epoches', default=10, type=int, help='total epoches to asm')
    parser.add_argument('--check_step', default=2, type=int, help='epoch steps to save model')
    parser.add_argument('--conf_thre', default=0.7, type=float, help='threshold to keep sl samples')
    parser.add_argument('--enable_al', default=True, type=bool, help='whether or not to use al process')
    parser.add_argument('--enable_sl', default=True, type=bool, help='whether or not to use ss process')

    args = parser.parse_args(params)
    return args


def get_learning_tag():
    tag = ''
    if args.enable_al and args.enable_sl:
        print('use al and sl samples')
        tag = 'as'
    elif args.enable_al:
        print('only use al samples')
        tag = 'al'
    elif args.enable_sl:
        print('only using sl samples is not meaningful!')
        tag = 'sl'
    return tag


# clear txt file to save sl/al idxs
with open('logs.txt', 'w') as f:
    f.truncate()


def update_asm_data_loader(tag, writer, epoch_idx):
    """
    infer on VOC2012 trainval data, update sl samples' bbox and label with model generated results
    """
    assert isinstance(writer, SummaryWriter)

    # detect on unlabel voc2012
    # unlabel_idxs, voc2012_anns are constant
    # what we want to observe: sl_idxs expand, al_idxs shrink
    print('detece on unlable data')
    sl_idxs, al_idxs, sl_anns, al_anns = detect_unlabel_imgs(model, unlabel_idxs, voc2012_anns, CONF_THRESH=args.conf_thre)
    writer.add_scalar('sl smaples', len(sl_idxs), global_step=epoch_idx)
    writer.add_scalar('al smaples', len(al_idxs), global_step=epoch_idx)

    with open('logs.txt', 'a') as f:  # append
        f.write('epoch: {}\n'.format(epoch_idx))
        f.write('sl_num: {}, al_num: {}\n'.format(len(sl_idxs), len(al_idxs)))
        f.write('sl: {}\n'.format(sl_idxs))
        f.write('al: {}\n\n'.format(al_idxs))

    asm_train_anns = voc2007_anns  # init with voc2007, will add active samples in voc2012 later
    if tag == 'as':
        asm_train_anns += (al_anns + sl_anns)  # todo: shuffle al/sl samples
    elif tag == 'al':  # todo: this may work better than as
        asm_train_anns += al_anns
    elif tag == 'sl':
        asm_train_anns += sl_anns

    dataset = VOC_Dataset(data=asm_train_anns, split=None,
                          transforms=get_transform(True))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=2,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    params = [
        '--gpus', '1',
        '--backbone', 'mobile',
        '--num_classes', '21',  # 20+bg
        '--pretrain_epoch', '4',
        '--conf_thre', '0.7',
        '--check_step', '2',
    ]
    args = parse_args(params)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # load pretrain model, to do active learning on VOC2012
    model = get_model(args.backbone, args.num_classes, self_pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
    ckpt = 'output/{}/{}_epoch_{}.pth'.format(args.backbone, args.backbone, args.pretrain_epoch)
    model.load_state_dict(torch.load(ckpt))
    model.cuda()
    print('load pretrain model on voc2007 done!')

    # prepare dataset
    voc2007_anns = load_data('VOC2007', split='trainval')  # 5011
    voc2012_anns = load_data('VOC2012', split='trainval')  # 11540
    voc2012_anns = voc2012_anns[:1000]  # only less samples
    unlabel_idxs = list(range(len(voc2012_anns)))

    asm_eval_anns = load_data('VOC2007', split='test')  # 4592
    asm_eval_anns = asm_eval_anns[:1000]
    dataset_eval = VOC_Dataset(data=asm_eval_anns, split=None,
                               transforms=get_transform(False))  # to tensor
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=collate_fn)
    print('load dataset done!')

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]  # conv1,conv2_x not update
    optimizer = torch.optim.SGD(params,
                                lr=0.004,
                                momentum=0.9, weight_decay=0.0005)
    # lr scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)  # 1/10
    # model save and tensor board
    model_save_dir = os.path.join('output', args.backbone)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    writer = SummaryWriter()

    # record pretrain model eval stats
    evaluate(model, data_loader_eval, writer, epoch_idx=0)

    # update data loader, allow al and sl data
    learn_tag = get_learning_tag()
    data_loader = update_asm_data_loader(learn_tag, writer, epoch_idx=0)

    # begin train again!
    for epoch in range(1, args.num_epoches + 1):
        model.train()
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, epoch,
                        print_freq=2,
                        writer=writer, begin_step=(epoch - 1) * len(data_loader))
        # update the learning rate
        lr_scheduler.step()
        if epoch % args.check_step == 0:
            evaluate(model, data_loader_eval, writer, epoch_idx=epoch)
            torch.save(model.state_dict(), os.path.join(model_save_dir,  # save model
                                                        '{}_{}_epoch_{}.pth'.format(args.backbone, learn_tag, epoch)))
            data_loader = update_asm_data_loader(learn_tag, writer, epoch_idx=epoch)
