from flask import (
    Blueprint, redirect, render_template, request, url_for, session, jsonify
)
from app.db import db, query_d_hits, parse_row_to_dict, query_one_userId, clear_auto_label
from app.db import find_project_by_name
from asm.demo import *
from asm.demo_classifier import *
from utils.app_utils import map_docker2host, hit_dict_tmp, hit_result_dict_tmp
from utils.box_utils import cvt_box_to_rect_fpts
from datetime import datetime
from utils.plt_utils import plt_bbox
from app.auth import login_required


bp = Blueprint('anno_auto', __name__)

# ajax 异步请求的时候 session 是不能在后台函数没有执行完成的时候更新的
# 这就导致辅助函数不能访问到没执行完成主函数的session的

data = {
    'total_num': 0,
    'cur_idx': 0,
    'progress': 0,
    'sl_num': 0,
    'al_num': 0,
    'sl_img_src': '',
    'al_img_src': '',
}

model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

choose_dataset_idx = None
project_list = None


# note: keep the useId!
hit_result_dict = hit_result_dict_tmp.copy()


def get_asm_models_by_dataset(dataset_name):
    asm_model_list = sorted(list(os.listdir(os.path.join(model_dir, dataset_name))))
    return asm_model_list


@bp.route('/ann_auto')
@login_required
def index():
    if session.get('choose_dataset') is None:
        return redirect('/datasets')
    else:
        global choose_dataset_idx, project_list
        choose_dataset_idx = session.get('choose_dataset_idx')
        project_list = session.get('project_list')
        return render_template(
            'ann_auto.html',
            choose_dataset=session.get('choose_dataset'),
            asm_model_list=get_asm_models_by_dataset(session.get('choose_dataset'))
        )


@bp.route('/progress', methods=['GET', 'POST'])
@login_required
def progress():
    return jsonify(data)


@bp.route('/clear_auto', methods=['GET', 'POST'])
@login_required
def clear_auto():
    dataset = request.values['dataset']
    if dataset is not None:
        if choose_dataset_idx is not None:
            clear_auto_label(project_id=project_list[choose_dataset_idx]['id'])
        else:
            project = find_project_by_name(session.get('choose_dataset'), session.get('project_list'))
            clear_auto_label(project_id=project['id'])
        print('clear all auto label data on', dataset)
    # reset asm status
    data = {
        'total_num': 0,
        'cur_idx': 0,
        'progress': 0,
        'sl_num': 0,
        'al_num': 0,
        'sl_img_src': '',
        'al_img_src': '',
    }
    return jsonify(data)


@bp.route('/auto_label', methods=['GET', 'POST'])
def auto_label():
    """
    insert row to d_hits_result + update status in d_hits
    """
    model, dataset = request.values['model'], request.values['dataset']
    if dataset is not None:
        if choose_dataset_idx is not None:
            class2names = project_list[choose_dataset_idx]['cats']
        else:
            project = find_project_by_name(session.get('choose_dataset'), session.get('project_list'))
            class2names = project['cats']

        # load model and corresponding detector
        if dataset == 'Cigar':
            asm_model = load_detection_model(dataset, model_name=model,
                                             num_classes=3)
        else:
            print("choose dataset idx:", choose_dataset_idx)
            print('project:', project_list[choose_dataset_idx])
            asm_model = load_detection_model(dataset, model_name=model,
                                             num_classes=project_list[choose_dataset_idx]['classes'])
        if model == 'yolov3-tiny':
            detect_func = yolov3_tiny_detect
        else:
            detect_func = faster_rcnn_detect

        # load fine-grained classification model
        if dataset == 'Cigar':
            load_classification_model()
            cigar_A_10_names = [
                'huanghelou_A', 'zhonghua_A', 'jiaozi_E', 'yuxi_A', 'wangguan_A', 'tianxiaxiu_A', 'huanghelou_E', 'jiaozi_C', 'jiaozi_F', 'huanghelou_C'
            ]
            cigar_a_10_names = [
                'yunyan_a', 'liqun_a', 'liqun_b', 'hehua_a', 'huangguashu_a', 'tianzi_c', 'huanghelou_d', 'furongwang_a', 'tianzi_a', '555_a'
            ]
            class2names = cigar_A_10_names + cigar_a_10_names  # extends return None

        data['cur_idx'] = data['sl_num'] = data['al_num'] = data['progress'] = 0
        data['sl_img_src'] = data['al_img_src'] = ''

        unlabeled_rows, data['total_num'] = query_d_hits(project_name=dataset, status='notDone')  # has no len
        print('total:', data['total_num'])

        userId = query_one_userId()

        for img_idx, row in enumerate(unlabeled_rows):
            print(img_idx)
            hit_dict = parse_row_to_dict(row, dict_template=hit_dict_tmp)
            img_path = hit_dict['data']
            img = Image.open(map_docker2host(img_path))  # todo: change to real path on nfs
            img_w, img_h = img.size

            # new hit_result_dict columns
            # id auto increment
            hit_result_dict['hitId'] = hit_dict['id']
            hit_result_dict['projectId'] = hit_dict['projectId']

            # do asm auto-label here
            boxes, labels = detect_func(asm_model, img)  # x1y1x2y2, label_id
            boxes = [list(map(int, box)) for box in boxes]
            labels = [int(label) for label in labels]

            # 1.insert row to d_hits_result
            if len(boxes) > 0:
                result = []
                for idx, (box, label) in enumerate(zip(boxes, labels)):
                    # can do fine-grain here
                    if dataset == 'Cigar':
                        roi = img.crop(box)
                        if label == 1:  # cigar_A
                            sub_label = classify_A(roi)  # [0,9]
                        elif label == 2:
                            sub_label = classify_a(roi) + len(cigar_A_10_names)  # add to one list
                        else:
                            sub_label = -1
                        labels[idx] = sub_label
                        box_info = {
                            "label": [class2names[labels[idx]]],
                            "shape": "rectangle",
                            "points": cvt_box_to_rect_fpts(box, img_w, img_h),
                            "notes": "",
                            "imageWidth": img_w,
                            "imageHeight": img_h
                        }
                    else:
                        labels[idx] = int(label - 1)
                        box_info = {
                            "label": [class2names[labels[idx]]],
                            "shape": "rectangle",
                            "points": cvt_box_to_rect_fpts(box, img_w, img_h),
                            "notes": "",
                            "imageWidth": img_w,
                            "imageHeight": img_h
                        }
                    result.append(box_info)

                    hit_result_dict['result'] = str(result).replace("'", '\\"')  # 转义字符，插入 mysql 使用
                    hit_result_dict['userId'] = userId
                    hit_result_dict['notes'] = 'auto-label'  # use to filter hard samples

                    # insert, create = update
                    hit_result_dict['created_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    hit_result_dict['updated_timestamp'] = hit_result_dict['created_timestamp']

            web_img_src = plt_bbox(img, boxes, labels, class2names)  # unlabeled 也可以这样

            if len(boxes) > 0:
                data['sl_img_src'] = web_img_src
                data['sl_num'] += 1
            else:
                data['al_img_src'] = web_img_src
                data['al_num'] += 1

            data['cur_idx'] = img_idx + 1
            data['progress'] = int(data['cur_idx'] / data['total_num'] * 100)

            if len(boxes) > 0:
                # value 单行, values 多行，但是速度 values 更快
                sql = "insert into d_hits_result " \
                      "(`hitId`, `projectId`, `result`, `userId`, `timeTakenToLabelInSec`, `notes`, `created_timestamp`, `updated_timestamp`) " \
                      "values ({},'{}','{}','{}',{},'{}','{}','{}')".format(hit_result_dict['hitId'],  # int
                                                                            hit_result_dict['projectId'],  # str
                                                                            hit_result_dict['result'],  # str
                                                                            hit_result_dict['userId'],  # str
                                                                            hit_result_dict['timeTakenToLabelInSec'],  # int
                                                                            hit_result_dict['notes'],  # str
                                                                            hit_result_dict['created_timestamp'],  # str
                                                                            hit_result_dict['updated_timestamp'])
                db.session.execute(sql)

                # 2.update status from 'notDone' to 'done' in d_hits
                sql = "update d_hits set status='sl' where id={}".format(hit_dict['id'])
                db.session.execute(sql)
                db.session.commit()

            else:
                sql = "update d_hits set status='al' where id={}".format(hit_dict['id'])
                db.session.execute(sql)
                db.session.commit()

        return jsonify(data)
    else:
        return redirect('/datasets')