import json
from collections import OrderedDict
import os
import random
import re
from utils.app_utils import map_docker2host
from utils.io_utils import dump_json
from utils.box_utils import cvt_rect_fpts_to_xywh, cvt_poly_fpts_to_center_xywh_angle


def filter_others(anns):
    return [ann for ann in anns if ann['label'][0] != 'others']


def filter_cigars(anns):
    cigar_anns = []
    for ann in anns:
        cat = ann['label'][0]
        if re.match('^.+_[A-Z]$', cat) or re.match('^.+_[a-z]$', cat):
            cigar_anns.append(ann)
    return cigar_anns


# process dataset
def create_dataset_from_sql_res(res):
    data = []
    cat_nums = {}
    for result in res:
        data.append({
            'img_id': result['img_id'],
            'anns': filter_cigars(json.loads(result['anns'])),  # original dataturks ann
            'path': result['path']
        })
        for ann in data[-1]['anns']:
            label = ann['label'][0]
            cat_nums[label] = cat_nums.get(label, 0) + 1  # default = 0
    # big->small dict
    cat_nums = OrderedDict(sorted(cat_nums.items(), key=lambda t: t[1], reverse=True))
    dataset = {
        'cat_nums': cat_nums,
        'data': data
    }
    return dataset


def create_dataset_from_dataturks_json(dataturks_json_path):
    data = []
    cat_nums = {}
    with open(dataturks_json_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for idx, line in enumerate(lines):
            product_dict = json.loads(line)
            data.append({
                'img_id': idx,
                'anns': filter_cigars(product_dict['annotation']),  # original dataturks ann
                'path': product_dict['content']
            })
            for ann in data[-1]['anns']:
                label = ann['label'][0]
                cat_nums[label] = cat_nums.get(label, 0) + 1  # default = 0
        # big->small dict
        cat_nums = OrderedDict(sorted(cat_nums.items(), key=lambda t: t[1], reverse=True))
        dataset = {
            'cat_nums': cat_nums,
            'data': data
        }
        return dataset


def get_subdict(ori_dict, sub_keys):
    sub_dict = OrderedDict()
    for key in sub_keys:
        sub_dict[key] = ori_dict[key]
    return sub_dict


def split_and_save_coco_dataset(dataset, dataset_dir, top_k=None, train_ratio=0.7, val_ratio=0.2):
    # filted all images and anns of selected cates
    filted_cats_num = dataset['cat_nums']
    filted_cats = list(filted_cats_num.keys())
    old_datas = dataset['data']
    if top_k is not None:
        filted_cats = filted_cats[:top_k]
        filted_cats_num = get_subdict(filted_cats_num, filted_cats)  # generate a sub dict from filter cats
        new_datas = []
        for old_data in old_datas:
            old_anns = old_data['anns']
            selected = False
            new_anns = []
            for old_ann in old_anns:
                label = old_ann['label'][0]
                if label in filted_cats:
                    new_anns.append(old_ann)
                    selected = True
            if selected:
                new_datas.append({
                    'img_id': old_data['img_id'],
                    'path': old_data['path'],
                    'anns': new_anns
                })
    else:
        new_datas = old_datas

    # random data
    random.shuffle(new_datas)
    total = len(new_datas)
    train_num, val_num = int(total * train_ratio), int(total * val_ratio)
    train_data = new_datas[:train_num]
    val_data = new_datas[train_num:train_num + val_num]
    test_data = new_datas[train_num + val_num:]

    # sava coco.json dataset
    save_coco_dataset(train_data, val_data, test_data, filted_cats, dataset_dir, use_prefix=True if top_k else False)

    return filted_cats, filted_cats_num, train_num, val_num, total - train_num - val_num


def save_coco_dataset(train_data, val_data, test_data, cats, dataset_dir, use_prefix=False):
    if use_prefix:
        prefix = '{}_'.format(len(cats))
    else:
        prefix = ''

    # cvt to coco
    dataset_name = os.path.basename(dataset_dir)
    train_coco = convert_to_coco(train_data, cats, info=dataset_name + ' train ' + prefix.replace('_', ''))
    val_coco = convert_to_coco(val_data, cats, info=dataset_name + ' val ' + prefix.replace('_', ''))
    test_coco = convert_to_coco(test_data, cats, info=dataset_name + ' test ' + prefix.replace('_', ''))

    # save
    dump_json(train_coco, out_path=os.path.join(dataset_dir, '{}{}_train.json').format(prefix, dataset_name))
    dump_json(val_coco, out_path=os.path.join(dataset_dir, '{}{}_val.json').format(prefix, dataset_name))
    dump_json(test_coco, out_path=os.path.join(dataset_dir, '{}{}_test.json').format(prefix, dataset_name))


# coco support choose cats
def convert_to_coco(data, cats, info='coco_dataset'):
    coco_dataset = {
        "info": info,
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories = {}
    cat_id = 0  # cat id according to num order
    for cat in cats:
        if re.match('.+_[A-Z]', cat):
            super_cat = '条装'
        elif re.match('.+_[a-z]', cat):
            super_cat = '包装'
        else:
            super_cat = ''
        category_coco = {
            "id": cat_id,
            "name": cat,
            "supercategory": super_cat,
        }
        categories[cat] = cat_id
        coco_dataset['categories'].append(category_coco)
        cat_id += 1
    # print(categories)

    ann_id = 0
    for result in data:
        anns = result['anns']
        if len(anns) == 0:
            continue
        # add image
        img_w, img_h = anns[0]['imageWidth'], anns[0]['imageHeight']
        image = {
            'coco_url': '',
            'data_captured': '',
            'file_name': map_docker2host(img_path=result['path']),
            'flickr_url': '',
            'id': result['img_id'],
            'height': img_h,
            'width': img_w,
            'license': 1,
        }
        coco_dataset['images'].append(image)

        # add anns
        for ann in anns:
            label = ann['label'][0]
            # add shape judgement
            rect_box, rect_angle = [], 0
            if ann['shape'] == 'rectangle':
                rect_box, rect_angle = cvt_rect_fpts_to_xywh(ann['points'], img_w, img_h), 0
            elif ann['shape'] == 'polygon':
                rect_box, rect_angle = cvt_poly_fpts_to_center_xywh_angle(ann['points'], img_w, img_h)
            anno_coco = {
                "segmentation": [],
                "area": [],
                "iscrowd": 0,
                "image_id": result['img_id'],
                "bbox": rect_box,
                "angle": rect_angle,
                "category_id": categories[label],
                "id": ann_id
            }
            coco_dataset['annotations'].append(anno_coco)
            ann_id += 1

    return coco_dataset


def cvt_echart_dict(project):
    # cvt project dict to echart json format dict
    root = {
        'name': project['name'],
        'images': project['train'] + project['valid'] + project['test'],
        'classes': project['classes'],
        'instances': sum(list(project['cats_num'].values())),
        'children': [],
    }
    # todo: super cat
    for cat, num in project['cats_num'].items():
        root['children'].append({
            'name': cat,
            'value': num
        })

    return root
