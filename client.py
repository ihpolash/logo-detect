"""
Client side code to perform a single API call to a tensorflow model up and running.
"""
import argparse
import json
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
import random
import string
from django.conf import settings
import pickle
import cv2
from imutils import paths
import pathlib
import urllib.request
from urllib.request import urlopen
from skimage import io
import shutil
import logoDetect.retrain_logo as retrain_logo


def get_random_alphaNumeric_string(stringLength=4):
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join((random.choice(lettersAndDigits) for i in range(stringLength)))


def get_logo_image_clssify(image):
    temp_image_name = get_random_alphaNumeric_string(4)
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # change this as you see fit
        image_path = default_storage.save(f"tmp/{temp_image_name}.jpg", ContentFile(image.read()))
        tmp_file = os.path.join(settings.MEDIA_URL, image_path)
        # print(image_path)
        image_data = default_storage.open(image_path).read()
        # Loads label file, strips off carriage return
        # Head Detector
        arr = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # img = cv2.imread(image_data)
        # print(img)
        height, width = img.shape[:2]
        # dim = (299, 299)
        face_data = cv2.imencode('.jpg', img)[1].tostring()

        # Loads label file, strips off carriage return

        with tf.io.gfile.GFile(f"tf_models/logo.txt", 'r') as fl:
            label_lines = [line.rstrip() for line in fl]
        # print("model loaded")

        # Unpersists graph from file
        with tf.io.gfile.GFile(f"tf_models/logo.pb", 'rb') as f:
            tf.compat.v1.reset_default_graph()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            pass
        res_arr = []
        scores = []

        final_predictions = tf_face_predictor(label_lines, face_data, 50, img)

        scores = final_predictions[0]
        id_string = final_predictions[1]
        res_arr.append({"company_name": id_string[np.argmax(scores)], "score": round(max(scores) * 100)})
        flag = True
        default_storage.delete(f'tmp/{temp_image_name}.jpg')
        if flag:
            logo_image_path = f"media/{res_arr[0]['company_name']}_{round(max(scores) * 100)}.jpg"
            if default_storage.exists(logo_image_path):
                default_storage.save(f"media/{res_arr[0]['company_name']}_{temp_image_name}_{round(max(scores) * 100)}.jpg",
                                        ContentFile(image_data))
                pass
            else:
                default_storage.save(f"media/{res_arr[0]['company_name']}_{round(max(scores) * 100)}.jpg",
                                        ContentFile(image_data))
            response = {"predictions": {
                "detection_classes": res_arr}}
            pass
        else:
            response = {"predictions": None, "message": "No match!"}
            pass
        return response
    except Exception as e:
        # print(e)
        default_storage.delete(f'tmp/{temp_image_name}.jpg')
        response = {"predictions": None, "message": "Low Prediction Score!"}
        return response

def url_to_image(url_link):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	# Your code where you can use urlopen
    with urlopen(url_link) as url:
        resp = url.read()
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


def get_logo_url_clssify(url):
    temp_image_name = get_random_alphaNumeric_string(4)
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # image_path = default_storage.save(f"tmp/{temp_image_name}.jpg", ContentFile(image.read()))
        # tmp_file = os.path.join(settings.MEDIA_URL, image_path)
        # # print(image_path)
        # image_data = default_storage.open(image_path).read()
        # # Loads label file, strips off carriage return
        # # Head Detector
        # arr = np.asarray(bytearray(image_data), dtype=np.uint8)
        img_data = requests.get(url).content
        img_path = f'media/image_name.jpg'
        with open(img_path, 'wb') as handler:
            handler.write(img_data)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        # dim = (299, 299)
        face_data = cv2.imencode('.jpg', img)[1].tostring()

        # Loads label file, strips off carriage return

        with tf.io.gfile.GFile(f"tf_models/logo.txt", 'r') as fl:
            label_lines = [line.rstrip() for line in fl]
        # print("model loaded")

        # Unpersists graph from file
        with tf.io.gfile.GFile(f"tf_models/logo.pb", 'rb') as f:
            tf.compat.v1.reset_default_graph()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            pass
        res_arr = []
        scores = []

        final_predictions = tf_face_predictor(label_lines, face_data, 50, img)

        scores = final_predictions[0]
        id_string = final_predictions[1]
        res_arr.append({"company_name": id_string[np.argmax(scores)], "score": round(max(scores) * 100)})
        flag = True
        # default_storage.delete(f'tmp/{temp_image_name}.jpg')
        if flag:
            response = {"predictions": {
                "detection_classes": res_arr}}
            pass
        else:
            response = {"predictions": None, "message": "No match found!"}
            pass
        return response
    except Exception as e:
        # print(e)
        # default_storage.delete(f'tmp/{temp_image_name}.jpg')
        response = {"predictions": None, "message": "Error found!"}
        return response



def tf_face_predictor(label_lines, image_data, threshold, im_resize):
    res_arr = []
    scores = []
    id_string = []
    with tf.compat.v1.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions_face = sess.run(softmax_tensor, \
                                    {'DecodeJpeg/contents:0': image_data})
        top_f = predictions_face[0].argsort()[-len(predictions_face[0]):][::-1]
        for node_id in top_f:
            face_string = label_lines[node_id]
            score_face = predictions_face[0][node_id]
            # face_string = label_lines[top_f[0]]
            # score_face = predictions_face[0][top_f[0]]
            # print(f'score: {score} , score_face: {score_face}')
            if round(score_face * 100) >= threshold:
                scores.append(score_face)
                id_string.append(face_string)
                pass
            else:
                pass

    return [scores, id_string]


def train_model(how_many_training_steps, testing_percentage, learning_rate, delete_checkpoint):
    dest_directory = f"train"
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    image_dir = f"train/dataset"
    output_graph = f"tf_models/logo.pb"
    output_labels = f"tf_models/logo.txt"
    output_graph_init = f"tf_models/logo_test.pb"
    output_labels_init = f"tf_models/logo_test.txt"
    summaries_dir = f"train/retrain_logs"
    # testing_percentage = 20
    validation_percentage = 10
    eval_step_interval = 1000
    train_batch_size = 100
    test_batch_size = -1
    validation_batch_size = -1
    bottleneck_dir = f"train/bottleneck"
    final_tensor_name = 'final_result'
    flip_left_right = False
    random_crop = 0
    random_scale = 0
    random_brightness = 0
    model_dir = f"train/inception"
    checkpoint_directory = f"checkpoint_dir"

    response = retrain_logo.main(image_dir, output_graph, output_labels, summaries_dir, how_many_training_steps, testing_percentage, validation_percentage, 
          eval_step_interval, train_batch_size, test_batch_size, validation_batch_size, bottleneck_dir, final_tensor_name, flip_left_right,
          random_crop, random_scale, random_brightness, model_dir, learning_rate,
          checkpoint_directory, delete_checkpoint, output_graph_init, output_labels_init)
    
    return response
