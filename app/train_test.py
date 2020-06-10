from flask import (
    Blueprint, render_template, request, session, jsonify, redirect
)
from pprint import pprint
from utils.io_utils import load_json, dump_json
from app.db import find_project_by_name
import os
import re
import traceback
import threading
import time
import shutil
from asm.detection.faster_rcnn.trainer import fast_rcnn_trainer
from asm.detection.yolo.trainer import yolo_trainer
from app.tensorboard import TensorBoardThread
from app.auth import login_required

bp = Blueprint('train_test', __name__)

model_tricks = {
    "faster_rcnn_res50": ['augment', 'cosine'],
    "faster_rcnn_mobile": ['augment', 'cosine'],
    "yolov3-tiny": ['augment', 'cosine', 'mixup', 'multiscale', 'aug_double']
}
model_list = list(model_tricks.keys())
eval_list = ['mAP', 'recall']
datasets_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
root_path = os.getcwd()
print(root_path)

train_status = {
    "training": False,
    "stage": "",
    "model": "",
    "tricks": [],
    "trained": 0,
    "cur_idx": 0,
    "total_num": 0,
    "msgs": {},
}

tb_path = ''
tb_process = None
trainer = None
tb_host = "10.214.211.205"
tb_port = 8805
tb_thread = None
starting_tb = False


def get_Aa_subs(cats):
    A_subs, a_subs = [], []
    for idx, cat in enumerate(cats):
        if re.match('.+_[A-Z]', cat):
            A_subs.append(idx)
        elif re.match('.+_[a-z]', cat):
            a_subs.append(idx)
    return A_subs, a_subs


def cvt_json_suepr(in_json, A_subs, a_subs):
    train_dict = load_json(in_json)
    for ann in train_dict['annotations']:
        if ann['category_id'] in A_subs:
            ann['category_id'] = 0
        elif ann['category_id'] in a_subs:
            ann['category_id'] = 1
    train_dict['categories'] = [
        {
            "id": 0,
            "name": "A"
        },
        {
            "id": 1,
            "name": "a"
        }]
    dump_json(train_dict, in_json.replace('.json', '_super.json'))


def cvt_Cigar_super(A_subs, a_subs):
    print("A_subs:", A_subs)
    print("a_subs:", a_subs)
    train_json = os.path.join(datasets_dir, 'Cigar', 'Cigar_train.json')
    val_json = os.path.join(datasets_dir, 'Cigar', 'Cigar_val.json')
    test_json = os.path.join(datasets_dir, 'Cigar', 'Cigar_test.json')

    cvt_json_suepr(train_json, A_subs, a_subs)
    cvt_json_suepr(val_json, A_subs, a_subs)
    cvt_json_suepr(test_json, A_subs, a_subs)


@bp.route('/train_status', methods=['GET', 'POST'])
@login_required
def status():
    return jsonify(train_status)


@bp.route('/train_test')
@login_required
def index():
    if session.get('choose_dataset') is None:
        return redirect('/datasets')
    else:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'models', session.get('choose_dataset'))
        pretrain_dict = {}
        for model in model_list:
            pretrain_list = []
            model_output_dir = os.path.join(output_dir, model)
            if os.path.exists(model_output_dir):
                for file in os.listdir(model_output_dir):
                    pretrain_list.append(file)
            pretrain_dict[model] = pretrain_list

        return render_template(
            'train_test.html',
            choose_dataset=session.get('choose_dataset'),
            model_tricks=model_tricks,
            pretrains=pretrain_dict,
            eval_list=eval_list,
            training=['false', 'true'][train_status['training']],
        )


def update_status(new_status):
    global tb_path
    train_status['training'] = (new_status['status'] == 'training')
    train_status['stage'] = new_status['stage']
    train_status['trained'] = new_status['epoch_trained']
    train_status['cur_idx'] = new_status['epoch_current']
    train_status['total_num'] = new_status['epoch_total']
    train_status['msgs'] = new_status['msgs']
    tb_path = new_status['tb_path']


def train_fast_rcnn_detector(backbone, tricks=[], epochs=10, pretrain_path=None, start_epoch=0, dataset_name=None):
    global trainer
    trainer = fast_rcnn_trainer()

    dataset_path = os.path.join(datasets_dir, 'Cigar')
    train_data = 'Cigar_train_super.json'
    # train_data = 'train_coco.json'
    valid_data = train_data.replace('train', 'val')
    test_data = train_data.replace('train', 'test')

    trainer.train(root_path,
                  dataset_path,
                  train_data,
                  valid_data,
                  test_data,
                  gpuid='0',
                  num_epochs=epochs,
                  tricks=tricks,
                  backbone=backbone,  # 'mobile'
                  num_classes=3,
                  check_step=5,
                  start_epoch=start_epoch,
                  dataset_name=dataset_name,
                  update_fn=update_status,
                  pretrained_path=pretrain_path)


def train_yolo_detector(tricks=[], epochs=10, pretrain_path=None, start_epoch=0, dataset_name=None):
    global trainer
    trainer = yolo_trainer()

    dataset_path = os.path.join(datasets_dir, 'Cigar')
    train_data = 'Cigar_train_super.json'
    # train_data = 'train_coco.json'
    valid_data = train_data.replace('train', 'val')
    test_data = train_data.replace('train', 'test')
    trainer.train(root_path,
                  dataset_path,
                  train_data,
                  valid_data,
                  test_data,
                  gpuid='1',
                  epochs=epochs,
                  start_epoch=start_epoch,
                  dataset_name=dataset_name,
                  tricks=tricks,
                  update_fn=update_status,
                  pretrained_path=pretrain_path,
                  debug=True)


# todo: add learning rate option
@bp.route('/start_train', methods=['GET', 'POST'])
@login_required
def start_train():
    if train_status['training']:
        return ""

    model_params = {
        'model': request.values['model'],
        'dataset': request.values['dataset'],
        'tricks': request.values['tricks'].split(','),
        'epoch': int(request.values['epoch']),
        'pretrain' : request.values['pretrain'],
        'start_epoch': int(request.values['start_epoch']),
    }
    pprint(model_params)

    # TODO: check the legal status of the model params

    global tb_path, tb_process
    tb_path = ""
    if tb_process is not None:
        tb_process.ternimate()
        tb_process = None

    train_status['training'] = True
    train_status['model'] = model_params['model']
    train_status['tricks'] = model_params['tricks']
    train_status['pretrain'] = model_params['pretrain']
    train_status['epoch'] = model_params['epoch']
    train_status['start_epoch'] = model_params['start_epoch']
    train_status['pretrain'] = model_params['pretrain']

    global datasets_dir
    # train/val/test data is coco format json
    if model_params['dataset'] == 'Cigar':
        train_super_json = os.path.join(datasets_dir, 'Cigar', 'Cigar_train_super.json')
        if not os.path.exists(train_super_json):
            project = find_project_by_name(session.get('choose_dataset'), session.get('project_list'))
            A_subs, a_subs = get_Aa_subs(project['cats'])
            cvt_Cigar_super(A_subs, a_subs)
        train_json = train_super_json
        val_json = train_json.replace('train', 'val')
        test_json = train_json.replace('train', 'test')
    else:
        train_json = os.path.join(datasets_dir, model_params['dataset'],
                                  '{}_train.json'.format(model_params['dataset']))
        val_json = os.path.join(datasets_dir, model_params['dataset'], '{}_val.json'.format(model_params['dataset']))
        test_json = os.path.join(datasets_dir, model_params['dataset'], '{}_test.json'.format(model_params['dataset']))

    if model_params["pretrain"] == 'None' or model_params["pretrain"] is None:
        pretrain = None
    else:
        pretrain = os.path.join(os.path.dirname(__file__),
                                '..', 'models', model_params["dataset"],
                                model_params['model'],
                                model_params["pretrain"])
    print(pretrain)
    try:
        if model_params['model'] == 'yolov3-tiny':
            train_yolo_detector(tricks=model_params['tricks'], epochs=model_params['epoch'],
                                dataset_name=model_params['dataset'],
                                start_epoch=model_params['start_epoch'], pretrain_path=pretrain)
        elif 'faster_rcnn' in model_params['model']:
            train_fast_rcnn_detector(backbone=model_params['model'],
                                     tricks=model_params['tricks'],
                                     epochs=model_params['epoch'],
                                     dataset_name=model_params['dataset'],
                                     start_epoch=model_params['start_epoch'],
                                     pretrain_path=pretrain)

    except BaseException as e:
        traceback.print_exc()
        train_status['msgs']['error'] = 'training error'
    finally:
        train_status['training'] = False
        return 'OK'


@bp.route('/train/replace_pretrain', methods=['GET', 'POST'])
@login_required
def replace_pretrain():
    print("replace model")
    try:
        model = request.values['model']
        dataset = request.values['dataset']
        src_path = os.path.join(root_path, 'output', dataset, model, 'best.pth')
        dst_path = os.path.join(root_path, 'models', dataset, model, 'best.pth')
        dst_dir = os.path.join(root_path, 'models', dataset, model)
        if not os.path.exists(src_path):
            raise FileNotFoundError
        if os.path.exists(dst_path):
            os.remove(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.move(src_path, dst_path)
        print(src_path)
    except Exception:
        traceback.print_exc()
        return 'NOT OK'
    return "OK"


@bp.route('/visualize_train', methods=['GET', 'POST'])
@login_required
def visualize_train():

    global tb_path, tb_port, starting_tb, tb_thread, tb_host
    if starting_tb:
        return "waiting"
    if tb_path == '':
        return ''

    # if the tb is starting, not return another
    starting_tb = True
    if tb_thread is not None:
        tb_thread.terminate()
        tb_thread = None
        time.sleep(3)

    # the port mybe already in use, so try another
    while True:
        signal = threading.Event()
        tb_thread = TensorBoardThread(signal, tb_path, tb_port)
        tb_thread.start()
        signal.wait()
        if tb_thread.wrong:
            tb_thread.terminate()
            tb_port += 1
        else:
            starting_tb = False
            break

    return jsonify(["http://{}:{}".format(tb_host, tb_port)])


@bp.route('/stop_train', methods=['GET', 'POST'])
@login_required
def stop_train():
    if trainer is not None:
        trainer.terminate()
        print('stop')
    return 'OK'
