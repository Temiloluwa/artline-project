import PIL.Image
import requests
import os
import re
import base64
from io import BytesIO
from fastai.vision import open_image
from matplotlib import pyplot as plt

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = PIL.Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def visualize_image(prediction):
    plt.imshow(prediction)
    plt.show()


def parse_base64(string_):
    base64_path = "data:image/jpeg;base64,"
    if string_.startswith(base64_path):
        string_ = re.sub(base64_path, "", string_)
        string_ =  bytes(string_, "UTF-8")
        return base64.b64decode(string_)
    else:
        return None


def image_size(img, size=500):
    """Resize Image and Maintain Aspect Ratio"""
    w, h = img.size
    aspect_ratio = w/h
    new_height = round(size/aspect_ratio)
    new_width = round(size * aspect_ratio)

    if new_width < new_height:
        return (new_width, size)
    else:
        return (size, new_height)


def preprocess_img(img_url, req_id):
    """ Prepocess Image for the model"""
    bytes_str = parse_base64(img_url)
    if not bytes_str:
        res = requests.get(img_url)
        if res.status_code != 200:
            try:
                res.raise_for_status()
            except Exception as e:
                return str(e)
        else:
            bytes_io = BytesIO(res.content)
    else:
        bytes_io = BytesIO(bytes_str)

        
    img = PIL.Image.open(bytes_io).convert("RGB")
    img = img.resize(image_size(img))
    img_path = f"cache/query_img_{req_id}.jpg"
    img.save(img_path)
    img = add_margin(img, 250, 250, 250, 250, (255, 255, 255))
    img.save(img_path, quality=95)
    img = open_image(img_path)
    os.remove(img_path)
    return img
            

def process_pred(pred):
    pred = pred.detach().cpu().numpy().transpose((1, 2, 0))
    pred = (pred - pred.min())/ (pred.max() - pred.min()) * 255
    pred = pred.astype("uint8")
    pred = PIL.Image.fromarray(pred)
    buffer = BytesIO()
    pred.save(buffer, format="JPEG")
    pred = base64.b64encode(buffer.getvalue())
    return pred


def convert_to_base64(img_path, del_img=True):
    """ Convert Image to base 64 """
    with open("img_path", "rb") as img:
        b64_img = base64.b64encode(img.read())
    if del_img:
        os.remove(img_path)
    return b64_img
