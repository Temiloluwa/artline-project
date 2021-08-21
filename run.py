from fastai import data_block
import torchvision.transforms as T
import os
import sys
import torch.nn as nn
import time
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import load_learner
from pathlib import Path
from utils import *
from flask import Flask, render_template, request
from flask_cors import CORS
from PIL import Image
from rq import Queue
from redis import Redis


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = upload_folder = os.path.join("static", "images")
pred_path = os.path.join(upload_folder, "pred_img.jpg")
query_path = os.path.join(upload_folder, "query_img.jpg")

cors = CORS(app, resources={r"/": {"origins": "*"}})

r = Redis(host='20.52.2.137', port=6379, db=0)
queue = Queue(connection=r)
                             
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


def predict(img_url, learner):
    img = preprocess_img(img_url)
    _,pred, _ = learner.predict(img)
    pred = pred.detach().cpu().numpy().transpose((1, 2, 0))
    pred = (pred - pred.min())/ (pred.max() - pred.min()) * 255
    pred = pred.astype("uint8")
    pred = Image.fromarray(pred)
    return pred

setattr(sys.modules["__main__"], 'FeatureLoss',  FeatureLoss)
learner = load_learner(Path("."), 'ArtLine_920.pkl')

def predict_img(query_url):
    print("Received Image")
    img = predict(query_url, learner)
    print("received prediction")
    img.save(pred_path)


def delete_imgs():
    #time.sleep(10)
    print("deleting images")
    os.remove(pred_path)
    os.remove(query_path)
    print("images deleted")



@app.route("/", methods=['GET', 'POST'])
def add_prediction_task():
    
    if 'query-url' not in request.form:
        if not os.path.exists(query_path) and not os.path.exists(pred_path):
            return render_template('index.html', p_image_path=".", q_image_path="#")

        elif os.path.exists(query_path) and not os.path.exists(pred_path):
            return render_template('index.html', p_image_path=\
                                    os.path.join(upload_folder,"still_process.jpeg"), q_image_path=query_path)
        else:
            #job = queue.enqueue(delete_imgs)
            return render_template('index.html',\
                p_image_path=pred_path, q_image_path=query_path)

    else:
        if os.path.exists(query_path):
            os.remove(query_path)
        
        if os.path.exists(pred_path):
            os.remove(pred_path)

        job = queue.enqueue(predict_img, request.form['query-url'])
        print(f"Job {job.id} added to queue  with {len(queue)} tasks in the queue")
        return render_template('index.html', p_image_path=\
                                os.path.join(upload_folder, "img_process.png"), q_image_path=query_path)
        
   
if __name__ == "__main__":
    app.run(host="0.0.0.0")
