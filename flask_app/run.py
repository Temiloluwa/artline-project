import re
from celery import Celery
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

REDIS_ADDR = 'redis://localhost:6379'
flask_app = Flask(__name__)
celery_app = Celery('celery_worker', broker=REDIS_ADDR, backend=REDIS_ADDR)
cors = CORS(flask_app, resources={r"/": {"origins": "*"}})


@flask_app.route("/", methods=['GET', 'POST'])
def add_prediction_task():
    if 'query-url' not in request.form:
        return render_template('index.html', p_image_path=".", q_image_path="#", task_id="")
    else:
        query_url = request.form['query-url']
        res = celery_app.send_task('tasks.predict_img', kwargs={"query_url": query_url})
        flask_app.logger.info(f"add task {res.backend}")
        response = {"task_id": res.id}
        return render_template('index.html', p_image_path=".", q_image_path=query_url, task_id=res.id)


@flask_app.route("/<task_id>")
def check_state(task_id):
    res = celery_app.AsyncResult(task_id, app=celery_app)
    flask_app.logger.info(f"state: {str(res.state)}")
    return jsonify(res.info if type(res.info) is not str else {'status': 200, 'msg':100})


@flask_app.route("/result/<task_id>")
def get_result(task_id):
    result = celery_app.AsyncResult(task_id).result
    result = "data:image/jpeg;base64," + result
    flask_app.logger.info(f"Obtained result of task: {task_id}")
    return jsonify({"status": 200, "msg": result})