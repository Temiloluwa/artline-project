import PIL.Image
import requests
from io import BytesIO
import os
from fastai.vision import open_image
from matplotlib import pyplot as plt
#from app import app

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = PIL.Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def preprocess_image(img_path, url=False):
    if url:
        response = requests.get(img_path)
        img = PIL.Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = PIL.Image.open(img_path).convert("RGB")
    #img_path = os.path.join(app.config['UPLOAD_FOLDER'], "input_image.jpg")
    img_path = os.path.join("static/images", "input_image.jpg")
    img = add_margin(img, 250, 250, 250, 250, (255, 255, 255))
    img.save(img_path, quality=95)
    img = open_image(img_path)
    return img

def visualize_image(prediction):
    plt.imshow(prediction)
    plt.show()
