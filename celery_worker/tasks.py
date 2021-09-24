import sys
from fastai.vision import *
from fastai.utils.mem import *
from fastai.vision import load_learner
from pathlib import Path
from celery import Celery
from celery.utils.log import get_task_logger
from model import FeatureLoss
from utils import *

REDIS_ADDR = 'redis://localhost:6379'
celery_app = Celery('tasks', broker=REDIS_ADDR, backend=REDIS_ADDR)
logger = get_task_logger(__name__)          
setattr(sys.modules["__main__"], 'FeatureLoss',  FeatureLoss)
learner = load_learner(Path("."), 'ArtLine_920.pkl')

@celery_app.task(bind=True)
def predict_img(self, query_url):
    req_id = self.request.id
    logger.info(f"Received Image with task : {req_id}")
    
    self.update_state(state="PROGRESS", meta={'status': 200, 'msg':20})
    img = preprocess_img(query_url, req_id)
    
    self.update_state(state="PROGRESS", meta={'status': 200, 'msg':40})
    _,pred, _ = learner.predict(img)
    
    self.update_state(state="PROGRESS", meta={'status': 200, 'msg':60})
    pred = process_pred(pred)
    logger.info("received prediction")
    self.update_state(state="PROGRESS", meta={'status': 200, 'msg':90})
    
    return pred.decode('utf-8')