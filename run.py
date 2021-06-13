import torchvision.transforms as T
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import load_learner
from pathlib import Path
from utils import *
from app import app
from flask import render_template, request
from PIL import Image

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


def predict(img_url, learner, url=False):
    img = preprocess_image(img_url, url)
    _,pred, _ = learner.predict(img)
    pred = pred.detach().cpu().numpy().transpose((1, 2, 0))
    pred = (pred - pred.min())/ (pred.max() - pred.min()) * 255
    pred = pred.astype("uint8")
    pred = Image.fromarray(pred)
    return pred

learner = load_learner(Path("."), 'checkpoint/ArtLine_920.pkl')

@app.route("/", methods=['GET', 'POST'])
def predict_image():
    if request.method == "POST":
        if 'query-image' not in request.files:
            return render_template('index.html', p_image_path=".", q_image_path="#")
        
        query_image = request.files['query-image']
        query_path = os.path.join(app.config['UPLOAD_FOLDER'],\
            query_image.filename)
        query_image.save(query_path)
        img = predict(query_path, learner, url=False)
        pred_path = os.path.join(app.config['UPLOAD_FOLDER'], \
            f"pred_{query_image.filename}")
        img.save(pred_path)
        return render_template('index.html',\
         p_image_path=pred_path, q_image_path=query_path)
    else:
        return render_template('index.html', p_image_path=".", q_image_path="#")

    
if __name__ == "__main__":
    app.run(debug=True)
