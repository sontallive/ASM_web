from flask import Flask, render_template
import os
import app.local_db


app = Flask(__name__)
app.config.from_object('config')

from app.auth import login_required

# load index page here!
@app.route('/')
@login_required
def index():
    # do this when start the web
    return render_template('index.html')


# todo: can't move to top of this file?
from . import datasets, anno_auto, train_test, auth


# register blueprint here
app.register_blueprint(datasets.bp)
app.register_blueprint(train_test.bp)
app.register_blueprint(anno_auto.bp)
app.register_blueprint(auth.bp)

local_db.init_app(app)


# load models
os.environ['CUDA_VISIBLE_DEVICES'] = app.config['GPU']
# load_detection_model(app.config['BACKBONE'], app.config['CP_EPOCH'])
