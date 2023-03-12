# Import des modules nécessaires
import pandas as pd
import numpy as np
import tensorflow as tf
import pathlib
import requests

from urllib import request
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO


def load_index_to_label(path: str = ''):
    index_to_class_label = pd.read_csv('.\\ressources\\final_500_targets.csv', sep = ';')
    return index_to_class_label

def load_model(path: str = '.\model\\'):
    # try:
    model = keras.models.load_model(path)
    return model
    # except OSError:
    # print('directory {} not found'.format(path))

def check_file(path):
    ret = False
    
    # Check file exist
    if pathlib.Path(path).is_file():
        ret = True
    else:
        r = request.urlopen(path)  # response
        if r.getcode() == 200:
            ret = True
    return ret

def image_to_array(upload_file):

    if upload_file.lower().startswith('http'):
        response = requests.get(upload_file)
        img = Image.open(BytesIO(response.content))
    else :
        img = Image.open(upload_file)
    
    img = img.resize(size = (160,160), resample = Image.NEAREST)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis = 0) 


def get_predictions(upload_file: str = "../images/zz106026.jpg", nb_preds: int=1):

    # Check file type
    if not upload_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) :
        raise TypeError('The file must be an image')
    
    if not check_file(upload_file):
        raise FileNotFoundError('file {} not found'.format(upload_file))

    # Check nb_preds value
    if not type(nb_preds) == int:
        raise TypeError('nb_preds must be an integer')
    
    img = image_to_array(upload_file)
    model = load_model()

    preds = model.predict(img)
    preds_sorted_proba = np.sort(preds)
    preds_sorted = np.argsort(preds, axis = -1)

    # create a list containing the class labels
    class_labels = load_index_to_label()

    # find the top 3 classes
    df_preds = pd.DataFrame({"name": class_labels.iloc[preds_sorted[0,-nb_preds:],1],
                             "proba": preds_sorted_proba[0, -nb_preds:]*100})
    df_preds = df_preds.sort_values('proba', axis=0, ascending=False)

    return df_preds

def get_wikilink(pred_name):
    # p = str(class_labels.iloc[preds_sorted[0,-1],1])
    link = 'https://fr.wikipedia.org/wiki/' + str(pred_name.lower().replace(' ', '_'))
    return link

predictions = get_predictions(upload_file = 'https://images.mushroomobserver.org/320/98097.jpg', nb_preds = 4)
print(predictions)