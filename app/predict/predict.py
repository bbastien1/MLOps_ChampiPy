# Import des modules nécessaires
import pandas as pd
import numpy as np
import pathlib
import requests
import os.path
import tensorflow as tf
import sys
import yaml

import mlflow.pyfunc
from urllib import request
from urllib.error import HTTPError
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from app.database.database import Database

def load_model(model_name: str="VGG16", stage: str = "Production"):

    mlruns_fld = os.path.realpath(os.path.join(SCRIPT_DIR, 'mlruns', 'models', model_name))

    for path, subdirs, files in os.walk(mlruns_fld):
        for name in files:    
            fullname = os.path.join(path, name)
            
            with open(fullname, 'r') as file:
                infos = yaml.safe_load(file)
                try:
                    if infos['current_stage'] == stage:
                        model_fld = infos['source']
                        print(model_fld)

                except KeyError:
                    # YAML dans model ne contient pas 'current_stage'
                    pass

    model_fld_split = model_fld.rsplit('mlruns')
    model_fld_fin = model_fld_split[1]
    model_fld_deb = os.path.realpath(os.path.join(SCRIPT_DIR, 'mlruns'))
    
    model_fld_final = model_fld_deb + model_fld_fin
    model_fld_final = os.path.realpath(model_fld_final)
    print("Model folder:", model_fld_final)
    model = mlflow.pyfunc.load_model(model_fld_final)

    return model


def check_file(path):
    ret = False
    
    # Check HTTP file exist
    if path.lower().startswith('http'):
        try:
            r = request.urlopen(path)  # response
            print("r :", r)
            if r.getcode() == 200:
                ret = True
        except HTTPError as err:
            if err.code == 404:
                ret = False
            else:
                print(str(err))
    # Check file exist
    else:
        if pathlib.Path(path).is_file():
            ret = True
        
    return ret


def image_to_array(upload_file):

    if upload_file.lower().startswith('http'):
        response = requests.get(upload_file)
        img = Image.open(BytesIO(response.content))
    else :
        img = Image.open(upload_file)
    
    img = img.resize(size = (120,120), resample = Image.NEAREST)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis = 0) 


def get_predictions(upload_file, nb_preds: int=1):

    # Check file type
    if not upload_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) :
        raise ValueError('The file must be an image')
    
    if not check_file(upload_file):
        raise FileNotFoundError('file {} not found'.format(upload_file))

    # Check nb_preds value
    if not type(nb_preds) == int:
        raise TypeError('nb_preds must be an integer')

    # Hide TF warnings
    tf.get_logger().setLevel('ERROR')

    img = image_to_array(upload_file)

    model = load_model()
    preds = model.predict(img)

    # create a list containing the class labels
    class_labels = get_classe_names()

    df_preds = pd.DataFrame({"name": class_labels,
                             "proba": preds[0]*100})
    
    # find the top X classes
    df_preds = df_preds.sort_values('proba', axis=0, ascending=False)
    df_preds = df_preds.iloc[:nb_preds]
    
    return df_preds.to_dict('records')


def get_accuracy():
    '''Return the accuracy evaluated of the trained model'''

    root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    
    model = load_model()
    eval_ds = get_eval_dataset(root_dir)

    loss0, accuracy0 = model.evaluate(eval_ds)
    
    return accuracy0


def get_eval_dataset(root_dir: str = ""):
    '''Return the evaluation dataset. Splited from get_accuracy to retrieve the classe names for predictions'''

    root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(root_dir, 'predict', 'images_eval')
    
    batch_size = 32
    img_height = 120
    img_width = 120

    # Evaluation Dataset
    eval_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    return eval_dataset


def get_classe_names(root_dir: str = ""):
    
    chpy_db = Database()
    return chpy_db.get_param('class_names')
