from flask import render_template, jsonify, session, Blueprint, redirect
from utils.dataset_utils import cvt_echart_dict
from app.db import query_dataset, clear_dataset
from app.auth import login_required

bp = Blueprint('datasets', __name__)

@bp.route('/datasets')  # all datasets
@bp.route('/datasets/<int:idx>', methods=['GET', 'POST'])  # dataset detail
@login_required
def index(idx=None):
    if idx is None:
        project_list = query_dataset()  # project is a dict with dataset info
        session['project_list'] = project_list
        return render_template(
            'datasets.html',
            project_list=session.get('project_list')
        )
    else:
        return jsonify(cvt_echart_dict(session.get('project_list')[idx - 1]))


@bp.route('/reload', methods=['GET', 'POST'])  # reload all datasets
@bp.route('/reload/<int:idx>', methods=['GET', 'POST'])  # only reload one dataset
@login_required
def reload_dataset(idx=None):
    if idx is None:
        clear_dataset()
        return redirect('/datasets')
    else:
        project_list = session.get('project_list')
        reload_project = project_list[int(idx - 1)]
        clear_dataset(dt_name=reload_project['name'])
        project_list[int(idx - 1)] = query_dataset(dt_name=reload_project['name'], project_idx=idx)
        session['project_list'] = project_list
        session['choose_dataset'] = reload_project['name']
        return session['choose_dataset']


@bp.route('/choose/<int:idx>', methods=['GET', 'POST'])
def choose_dataset(idx=None):
    if idx > 0:
        choose_dataset_idx = idx - 1
        session['choose_dataset_idx'] = choose_dataset_idx
        session['choose_dataset'] = session.get('project_list')[choose_dataset_idx]['name']
        return session['choose_dataset']