import os

SECRET_KEY = 'vipa-asm'
SESSION_TYPE = 'filesystem'
# PERMANENT_SESSION_LIFETIME = timedelta(days=1)

# docker path map
HOST_UPLOADS = '/nfs/xs/docker/vipaturks/uploads'

# database
host = '10.214.211.205'
port = 3316
db = 'hope'
username, password = 'dataturks', '12345'
# must +pymysql
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(username, password, host, port, db)
SQLALCHEMY_COMMIT_TEARDOWN = True  # auto commit db changes
SQLALCHEMY_TRACK_MODIFICATIONS = False
# SQLALCHEMY_ECHO = True  # 打印查询语句到 console

# 本地sqllite文件地址
LOCAL_DATABASE = os.path.join('app/instance', 'app.sqlite')

# model
GPU = '0'
BACKBONE = 'mobile'
CP_EPOCH = 10  # checkpoint epoch
