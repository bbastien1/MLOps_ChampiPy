# Import des modules nécessaires

from tensorflow import keras
from PIL import Image
import pandas as pd
import numpy as np
#import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_index_to_label(path: str = ''):
        index_to_class_label = pd.read_csv('.\\ressources\\final_500_targets.csv', sep = ';')
        return index_to_class_label

def load_model(path: str = '.\model\\'):
    model = keras.models.load_model(path)
    return model

# def predict(img):   
#     prediction = model.predict(img)
#     return prediction


def image_to_array(upload_file):
    img = Image.open(upload_file)
    img = img.resize(size = (160,160), resample = Image.NEAREST)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis = 0) 
    

def get_predictions(upload_file: str = "../images/106026.jpg", nb_preds: int=3):
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

    # print("Top 3 des espèces les plus probables :")
    # print(df_top3)

    # # Lien wikipédia
    # p = str(class_labels.iloc[preds_sorted[0,-1],1])
    # link = 'https://fr.wikipedia.org/wiki/' + str(p.lower().replace(' ', '_'))
    # print("Lien wikipédia vers l'espèce la plus probable", link)
    return df_preds

predictions = get_predictions(nb_preds = 4)
print(predictions)