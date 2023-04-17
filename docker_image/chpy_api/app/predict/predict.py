# Import des modules nécessaires
import pandas as pd
import numpy as np
import pathlib
import requests

from urllib import request
from urllib.error import HTTPError
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO


def load_index_to_label():
    file = '.\\predict\\target_name.csv'
    
    if not check_file(file):
        raise FileNotFoundError('file {} not found'.format(file))
    
    index_to_class_label = pd.read_csv(file, sep = ';')
    
    return index_to_class_label


def load_model(path: str = '.\\predict\\model\\'):
    model = keras.models.load_model(path)
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
    
    img = img.resize(size = (160,160), resample = Image.NEAREST)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis = 0) 


def get_predictions(upload_file, nb_preds: int=1):

    print("upload file :", upload_file)

    # Check file type
    if not upload_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) :
        raise ValueError('The file must be an image')
    
    if not check_file(upload_file):
        raise FileNotFoundError('file {} not found'.format(upload_file))

    # Check nb_preds value
    if not type(nb_preds) == int:
        raise TypeError('nb_preds must be an integer')
    print("upload OK")
    img = image_to_array(upload_file)
    model = load_model()
    print("load_model OK")
    preds = model.predict(img)
    print("preds OK")
    preds_sorted_proba = np.sort(preds)
    preds_sorted = np.argsort(preds, axis = -1)
    print("preds_sorted OK")
    # create a list containing the class labels
    class_labels = load_index_to_label()
    print("class_label OK")
    # find the top X classes
    df_preds = pd.DataFrame({"name": class_labels.iloc[preds_sorted[0,-nb_preds:],1],
                             "proba": preds_sorted_proba[0, -nb_preds:]*100})
    df_preds = df_preds.sort_values('proba', axis=0, ascending=False)

    #return df_preds.to_json(orient="records")
    return df_preds.to_dict('records')