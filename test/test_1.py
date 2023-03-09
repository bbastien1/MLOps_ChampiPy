# Import des modules nécessaires

from tensorflow import keras
from PIL import Image
import pandas as pd
import numpy as np
#import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_index_to_label(path: str = ''):
        index_to_class_label = pd.read_csv('.\\ressources\\final_500_targets.csv', sep = ';')
        return index_to_class_label

def load_model(path: str = '.\model\\'):
    model = keras.models.load_model(path)
    return model

def predict(img):   
    prediction = model.predict(img)
    return prediction


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

# Contenu de la deuxième page
upload_file = "../images/164.jpg"
    
if upload_file:

    img = Image.open(upload_file)
    img = img.resize(size = (160,160), resample = Image.NEAREST)
    img_array = image.img_to_array(img)
    x = np.expand_dims(img_array, axis = 0)

    model = load_model()
    preds = model.predict(x)
    preds_sorted_proba = np.sort(preds)
    preds_sorted = np.argsort(preds, axis = -1, order = None)

    # create a list containing the class labels
    class_labels = load_index_to_label()

    # find the index of the class with maximum score
    pred = np.argmax(preds, axis = -1)

    # print the label of the class with maximum score
    print("D'après notre modèle, votre champignon appartient à l'espèce : {}".format(class_labels.iloc[pred[0],1]))

    # print the label of the class with maximum score
    print("Notre prédiction de l'espèce")
    print("D'après notre modèle, votre champignon à une chance d'appartenir à une de ces espèces : ")

    print("Espèce la plus probable {}".format(class_labels.iloc[preds_sorted[0,-1],1]))
    print("Probabilité (%) {}".format(int(preds_sorted_proba[0, -1]*100)))
    
    # Lien wikipédia
    p = str(class_labels.iloc[preds_sorted[0,-1],1])
    link = 'https://fr.wikipedia.org/wiki/' + str(p.lower().replace(' ', '_'))
    print("Lien wikipédia vers l'espèce la plus probable", link)
    
    print("2e possibilité", class_labels.iloc[preds_sorted[0,-2],1])
    print("Probabilité (%)", int(preds_sorted_proba[0, -2]*100))
    
    print("3e possibilité", class_labels.iloc[preds_sorted[0,-3],1])
    print("Probabilité (%)", int(preds_sorted_proba[0, -3]*100))









