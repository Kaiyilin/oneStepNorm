from email import message
from flask import (
    Flask, 
    render_template, 
    request,
    redirect
    )
from sklearn.feature_extraction import img_to_graph
from ops.deploy import (
    obtain_med_image,
    pre_process,
    output_process
    )
from nilearn.image import resample_img
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib 
import tensorflow as tf 

PAD_SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./Models")
parser.add_argument("--output_path", type=str, default="./Outputs/")
parser.add_argument("--data_path", type=str, default="./data_raw")
args = vars(parser.parse_args())

# initialise an Flask app
app = Flask(__name__)

# load a deep learning model
model = tf.keras.models.load_model(args["model_path"])

# load image header
src_info = nib.load("./hdraffine.nii")

# validate output directory
try:
    os.mkdir(args["output_path"])
except FileExistsError as e:
    print(e)


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/', methods=["GET", "POST"])
def predict():
    """
    Put the work you'd like to complete here
    """
    img_file = request.files["img_file"]
    # validate the file format
    if not img_file.filename.endswith(".nii"):
        return render_template(
            "index.html",
            message="Wrong file format, the file should be an NIFTI (.nii) image")
    
    else:
        img_array = obtain_med_image(path=args["data_path"], image=img_file.filename)
        shape_validation = all(i <= PAD_SIZE for i in img_array.shape)

    if 2 < img_array.ndim < 5 and shape_validation:
        padded_img_array = pre_process(img_array, pad_size=PAD_SIZE, mode=1)
        product = model.predict(padded_img_array)
        clipped_img = output_process(product, src_info)
        nib.save(clipped_img, os.path.join(args["output_path"], f"Normed_{img_file.filename}"))

        return render_template(
            'index.html', 
            img_file=os.path.join(args["output_path"], f"Normed_{img_file.filename}")
            )
    
    else:
        message = f"The current file with size {img_array.shape} does not support by the model"
        return render_template(
            'index.html',
            message=message)


if __name__ == '__main__':
    print("\t* Loading Keras model and Flask starting server...\n"
        "please wait until server has fully started")
    app.run(port=5000, debug=True)