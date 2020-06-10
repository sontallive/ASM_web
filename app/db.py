from flask_sqlalchemy import SQLAlchemy
import json
import os
import shutil
from utils.dataset_utils import create_dataset_from_sql_res, split_and_save_coco_dataset
from utils.io_utils import dump_json, load_json
from app import app
import hashlib

# load db
db = SQLAlchemy(app)
task_type = {
    'rectangle': 'Detection',
    'polygon': 'Segmentaion'
}

data_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../datasets')

def sha256(text):
    encrypt = hashlib.sha256()
    encrypt.update(text.encode())
    res = encrypt.hexdigest()
    return res

def query_user(email, password):
    """
    :param email: the email of user
    :param password: the password of user
    :return: a login json, full information if login successfully
    """
    userinfo = {
        'login': False,
        'id': '',
        'firstName': '',
        'secondName': '',
        'password': ''
    }
    sql = "select id, firstName, secondName, password from d_users where email = '{}'"
    user = db.session.execute(sql.format(email)).fetchone()
    if user is not None and user['password'] == sha256(password):
        userinfo['login'] = True
        userinfo['id'] = user['id']
        userinfo['firstName'] = user['firstName']
        userinfo['secondName'] = user['secondName']
        userinfo['password'] = user['password']

    return userinfo

def query_user_by_id(userid):
    """
    :param userid: the id of user
    :return: a user info json, full information of user
    """
    userinfo = {
        'id': '',
        'email': '',
        'firstName': '',
        'secondName': '',
        'password': ''
    }
    sql = "select id, email, firstName, secondName, password from d_users where id = '{}'"
    user = db.session.execute(sql.format(userid)).fetchone()

    userinfo['login'] = True
    userinfo['id'] = user['id']
    userinfo['email'] = user['email']
    userinfo['firstName'] = user['firstName']
    userinfo['secondName'] = user['secondName']
    userinfo['password'] = user['password']

    return userinfo

def register_user():

    pass

def parse_row_to_dict(row, dict_template):
    """
    :param row: query result: d_hits, d_hits_result
    :param dict_template: hit_dict, hit_result_dict
    :return:
    """
    dict_row = dict_template.copy()
    for idx, (key, _) in enumerate(dict_row.items()):  # items 结果相对字典还是有序的
        dict_row[key] = row[idx]
    return dict_row

def query_d_hits(project_name, status=None):
    """
    :param project_name: voc, Retail Project Dataset
    :param status: 'done', 'notDone'
    :return:
    """
    sql = "select * from d_hits " \
          "where projectId in (select id from d_projects where name='{}')".format(project_name)
    if status:
        sql += "and status='{}'".format(status)
    res = db.session.execute(sql)
    # 总数
    sql = sql.replace('*', 'count(*)')
    total_num = db.session.execute(sql)
    # 以循环的方式解析结果, yield, 一输出就没了
    # for row in res:
    #     print(row)
    #     pprint(parse_row_to_dict(row, dict_template=hit_dict_tmp))
    return res, total_num.next()[0]

def query_d_hits_result(project_name):
    sql = "select * from d_hits_result " \
          "where projectId in (select id from d_projects where name='{}')".format(project_name)
    res = db.session.execute(sql)
    return res

def query_one_userId():
    sql = "select id from d_users limit 0,1"
    res = db.session.execute(sql)
    return res.next()[0]

def clear_cigars():
    # find cigars rows
    projectId = '2c9180836e7e50c2016e8136924a0000'
    sql = "select hitId from d_hits_result  where projectId='{}'".format(projectId)
    hitIds = db.session.execute(sql)

    for hid in hitIds:
        hid = hid[0]
        # clear d_hits
        sql = "delete from d_hits where id='{}'".format(hid)
        db.session.execute(sql)

    # clear d_hits_result
    sql = "delete from d_hits_result where projectId='{}'".format(projectId)
    db.session.execute(sql)
    db.session.commit()


def clear_auto_label(project_id):
    # find auto-label rows
    sql = "select hitId from d_hits_result where projectId='{}' and notes='auto-label'".format(project_id)
    hitIds = db.session.execute(sql)

    for hid in hitIds:
        hid = hid[0]
        # update d_hits status
        sql = "update d_hits set status='notDone' where id={}".format(hid)
        db.session.execute(sql)
        # clear d_hit_results
        sql = "delete from d_hits_result where hitId={}".format(hid)
        db.session.execute(sql)
        db.session.commit()


def update_d_hits_data():
    """
    copy data from '/nfs/xs/retail/uploads' to '/nfs/xs/docker/vipaturks/uploads'
    as projectId are different, the data column containing img path should change
    """
    old_projectId = '2c9180826d47a650016d5e359eaf0004'
    new_projectId = '2c9180836e7e50c2016e8136924a0000'

    sql = "select data from d_hits where projectId='{}'".format(new_projectId)
    img_paths = db.session.execute(sql)

    for p in img_paths:
        old_path = p[0]
        new_path = old_path.replace(old_projectId, new_projectId)
        sql = "update d_hits set data='{}' where data='{}'".format(new_path, old_path)
        db.session.execute(sql)
        db.session.commit()


def update_d_hits_result_userId():
    old_userId = '2c9180836e7e50c2016e8136924a0000'
    new_userId = 'x4o52HKK3HJbrLq4BI3cwEA5sxE4'

    sql = "update d_hits_result set userId='{}' where userId='{}'".format(new_userId, old_userId)
    db.session.execute(sql)
    db.session.commit()


def parse_projects(row):
    taskRules = json.loads(row[3])  # str->dict, 中文
    cats = taskRules['tags'].replace(' ', '').split(',')
    print(cats)
    print(taskRules)
    task = task_type.get(taskRules.get('defaultShape'))
    return {
        'id': row[0],
        'name': row[1],
        'taskType': task,
        'classes': len(cats),
        'cats': cats,
    }


def query_dataset(dt_name=None, project_idx=None):
    if dt_name is None:  # query all
        sql = "select name from d_projects"
        # project_names = db.session.execute(sql)
        # project_names = ['Cigar', 'OCR', 'VOC']
        project_names = ['Cigar']
        project_list = []
        for idx, p_name in enumerate(project_names):
            # project = project_detail(p_name[0])  # will save json here
            project = project_detail(p_name)  # will save json here
            project['idx'] = idx + 1  # add idx
            project_list.append(project)
        return project_list
    else:
        project = project_detail(dt_name)
        project['idx'] = project_idx
        return project


def clear_dataset(dt_name=None):
    if dt_name is None:  # clear all
        for dt in os.listdir(data_root):
            shutil.rmtree(os.path.join(data_root, dt))
            print('clear', dt)
    else:
        shutil.rmtree(os.path.join(data_root, dt_name))
        print('clear', dt_name)


def find_project_by_name(name, project_list):
    print("project list:", project_list)
    for project in project_list:
        if project['name'] == name:
            return project


def project_detail(project_name, top_k=None):
    dataset_dir = os.path.join(data_root, project_name)
    print('dataset_dir', dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    prefix = '{}_'.format(top_k) if top_k else ''
    project_cfg_path = os.path.join(dataset_dir, '{}{}_cfg.json'.format(prefix, project_name))

    if os.path.exists(project_cfg_path):  # has dataset_file
        print('read {} from json!'.format(project_name))
        project = load_json(project_cfg_path)
    else:
        print('read {} from mysql!'.format(project_name))
        # read project cfgs
        sql = "select id, name, taskType, taskRules from d_projects where name='{}'".format(project_name)
        res = db.session.execute(sql)
        project = res.next()
        project = parse_projects(project)

        # read data from mysql
        # todo: store the largest hitId for a project, better for update dataset
        sql = "select d_hits.id as img_id, d_hits.data as path, d_hits_result.result as anns from d_hits, d_hits_result " \
              "where d_hits.projectId='{}' and d_hits.id=d_hits_result.hitId and d_hits.status='done'".format(project['id'])
        res = db.session.execute(sql)
        dataset = create_dataset_from_sql_res(res)
        filted_cats, filted_cats_num, train_num, val_num, test_num = split_and_save_coco_dataset(dataset, dataset_dir, top_k)

        # update project
        project['cats'] = filted_cats
        project['cats_num'] = filted_cats_num
        project['classes'] = len(filted_cats)
        project['train'] = train_num
        project['valid'] = val_num
        project['test'] = test_num

        dump_json(project, out_path=os.path.join(dataset_dir, '{}{}_cfg.json'.format(prefix, project_name)))

    return project


if __name__ == '__main__':
    # projects = query_all_datasets()
    pass
