import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, current_app, make_response
)
from werkzeug.security import check_password_hash, generate_password_hash
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from app.local_db import get_db
from app.db import query_user, query_user_by_id
import requests  # also from flask package
import json

bp = Blueprint('auth', __name__, url_prefix='/auth')


def create_token(api_user):
    """
    生成token
    :param api_user:用户id
    :return: token
    """
    s = Serializer(current_app.config['SECRET_KEY'], expires_in=3600)
    token = s.dumps({'id': api_user}).decode("ascii")
    return token


def verify_token(token):
    """
    校验token
    :param token:
    :return: 用户信息 or None
    """
    # 参数为私有秘钥，跟上面方法的秘钥保持一致
    s = Serializer(current_app.config["SECRET_KEY"])
    try:
        # 转换为字典
        data = s.loads(token)
    except Exception:
        return None

    return data['id']


@bp.before_app_request
def load_logged_in_user():
    token = request.cookies.get("auth")
    g.auth = token
    # print("get token:",token)


def login_required(view):
    """
    a wrapper of login required pages or apis
    """
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        # g: global
        if g.auth is None:
            return redirect(url_for('auth.login'))
        userid = verify_token(g.auth)
        if userid is None:
            return redirect(url_for('auth.login'))
        else:
            g.userid = userid
            print("auth success,userid is:", userid)
        return view(**kwargs)
    return wrapped_view


@bp.route('/register', methods=('GET', 'POST'))
def register():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        firstName = request.form['firstName']
        secondName = request.form['secondName']
        password = request.form['password']

        if not email:
            error = 'Email is required.'
        elif not password:
            error = 'Password is required.'
        else:
            # simulate POST request to create user in dataturks
            data = {
                'firstName': firstName,
                'secondName': secondName,
                'authType': 'emailSignUp',  # use email to sign up
                "email": email
            }
            headers = {
                'Content-Type': 'application/json',
                'password': password
            }
            # todo: dataturks base url
            url = "http://10.214.211.205/dataturks/createUserWithPassword"  # can test wrong url
            # return <requests.Response>
            rep = requests.post(url=url, data=json.dumps(data), headers=headers)  # <Response [200]>, 400, ...
            # error = rep.raise_for_status()  # will raise error if http error, can't process in this way

            if rep.status_code == 404:  # int
                error = 'url 404 not found'
            else:
                status = json.loads(rep.text)  # property func, str -> dict
                # 200 {'id': 'eyvDoRF0HSXA12WIAFF3oTJNLe3a', 'token': 'i0YeDQT7Vx68BlnEuTkN6K1zbLz4At01qQfTFYOCB35k8mnG8eJ6ChZYecbkOwI2'}
                # 400 {'code': 400, 'message': 'User with the given email already exists'}, error msg from dataturks
                if status.get('code', -1) != -1:  # if status return code
                    error = status.get('message', 'message is None')

        if error is None:  # now can login
            return redirect(url_for('auth.login'))

    return render_template('auth/register.html', error=error)


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = query_user(username, password)

        if not user['login']:
            error = 'Login failed.'
        else:
            token = create_token(user['id'])
            redir = redirect(url_for('index'))
            response = make_response(redir)
            # set the response token
            response.set_cookie("auth", token, max_age=3600)
            return response

    return render_template('auth/login.html')


@bp.route('/logout', methods=('GET', 'POST'))
def logout():
    redir = redirect(url_for('auth.login'))
    resp = make_response(redir)
    resp.delete_cookie('auth') # clear the token from cookie
    return resp


@bp.route('/login_dataturks', methods=('GET', 'POST'))
@login_required
def login_dataturks():
    # get the logined user
    userid = g.userid
    user = query_user_by_id(userid)
    headers = {
        'Content-Type': 'application/json', 'email': user['email'],
        'encryptedPassword': user['password']
    }
    print('email', user['email'])
    print('password', user['password'])

    url = "http://10.214.211.205/dataturks/loginByEncrypted"
    rep = requests.post(url=url, data=None, headers=headers) # post the backend of dataturks to get the login auth
    print('rep is', rep.text)
    logininfo = json.loads(rep.text)

    return redirect("http://10.214.211.205/autologin.html?id={}&token={}".format(logininfo['id'], logininfo['token']))
