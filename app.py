import os
import keras
import numpy as np
import pickle
# import config as cfg

from deepModel import model
from getIDs import topIDs
import cv2

from flask import Flask, request, render_template, send_from_directory

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model = model()

with open('cosine_train_vectors.pickle', 'rb') as f:
    train_vectors = pickle.load(f)

with open('data_vectors.pickle', 'rb') as f:
    data = pickle.load(f)

app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():

    upload_dir = os.path.join(APP_ROOT, "uploads/")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    for img in request.files.getlist("file"):
	    img_name = img.filename
	    destination = "/".join([upload_dir, img_name])
	    img.save(destination)

    result = topIDs(model, train_vectors, os.path.join(upload_dir, img_name))

    for i in range(len(result)):
        cv2.imwrite(f'static/result_images/{i}.png', data[result[i]]*255)

    result_final = []

    for i in range(len(result)):
        result_final.append(f"result_images/{i}.png")

    return render_template("results.html", image_name=img_name, result_paths=result_final)

#Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("uploads", filename)


if __name__ == "__main__":
	app.run(port=5000, debug=True)