import numpy as np
import requests
import datetime
import pickle
import os
import gridfs

from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient, errors, timeout
from bson.binary import Binary
from urllib.parse import urlparse

class Database:
    """
    """

    client = None
    DATABASE = None
    fs = None

    def __init__(self):
        CONNECTION_STRING = "mongodb+srv://champipy:CCeD3AyOtqxvw2iJ@cluster0.iul9opn.mongodb.net/champipy_db"
        Database.client = MongoClient(CONNECTION_STRING)
        Database.DATABASE = Database.client.champipy_db
        Database.fs = gridfs.GridFS(Database.DATABASE, 'img_store')

    def check_db_connex(self):
        try:
            with timeout(3):
                not_used = Database.client.server_info()
                return True
        except errors.ServerSelectionTimeoutError:
            return False


    def get_user(self, user:str):
        ret = None
        if user :
            collection_users = Database.DATABASE["users"]
            user_query = { "username": user }
            user = collection_users.find_one(user_query)
            if user != None :
                ret = user["username"]    
        return ret


    def get_user_pwd(self, user:str):
        collection_users = Database.DATABASE["users"]
        user_query = { "username": user }
        user = collection_users.find_one(user_query)
        return user["hashed_password"]
    

    def add_image_to_db(user:str, file:str, classname:str, name:str):
        
        if not Database.fs.exists(filename=name):

            if file.lower().startswith('http'):
                response = requests.get(file)
                contents = response.content
            else :
                with open(file, 'rb') as f:
                    contents = f.read()

            a = Database.fs.put(contents, filename=name, classname=classname, user=user)
        else:
            print("Image already saved")


    def save_prediction(self, user:str, file:str, results):

        filename = urlparse(file)
        filename = os.path.basename(filename.path)
        
        prediction = {
            'user': user,
            'url': file,
            'filename': filename,
            'datetime':datetime.datetime.now(),
            'results': results
        }

        Database.DATABASE["predictions"].insert_one(prediction)

        # Store image on DB if proba >= 70 %
        if results[0]['proba'] >= 70:
            Database.add_image_to_db(user=user, file=file, classname=results[0]['name'], name=filename)



    def get_param(self, param_name:str):
        collection_param = Database.DATABASE["parameters"]
        query = { "param_name": param_name }
        document = collection_param.find_one(query)
        return document["param_value"]
    

    def is_already_predicted(file):
        ret = None
        if Database.fs.exists(filename=file):
            collection_preds = Database.DATABASE["predictions"]
            query = { "filename": file }
            prediction = collection_preds.find_one(query)
            if prediction != None:
                ret = prediction['results']

        return ret
    