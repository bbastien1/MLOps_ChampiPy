import sys
import os
import json

from fastapi import Depends, FastAPI, HTTPException, status, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from urllib.parse import urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import app.predict.finetune as ft
from app.predict.predict import get_predictions, get_accuracy, get_model_date
from app.database.database import Database


# define Python user-defined exceptions
class DbConnexError(BaseException):
    "Raised when database connection failed"
    pass


api = FastAPI(
    title='MLOps_ChampiPy',
    description="API du projet ChampiPy dans le cadre de la formation MLOps",
    version="0.1.0",

    openapi_tags=[
    {
        'name': 'home',
        'description': 'Fonctions générales'
    },
    {
        'name': 'predictions',
        'description': 'Fonctions utilisées pour obtenir des prédictions'
    },
    {
        'name': 'supervise',
        'description': 'Fonctions utilisées pour superviser le modèle'
    }
]
)


security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
chpy_db = Database()

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    
    if chpy_db.check_db_connex():
        if not(chpy_db.get_user(username)) or \
                not(pwd_context.verify(credentials.password, chpy_db.get_user_pwd(username))):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        ret={'username': credentials.username, 
             'is_admin': chpy_db.is_user_role(username, 'admin')}
        return ret
    else:
        raise DbConnexError("Database connection failed")


@api.get('/status', tags=['home'])
async def get_api_status():
    '''
    Retourne True si l'API est fonctionnelle
    '''
    return True


@api.get('/dbconnex', tags=['home'])
async def get_db_connex_status():
    '''
    Retourne True si l'API arrive à se connecter à la BDD
    '''
    return chpy_db.check_db_connex()


@api.get('/user', tags=['home'])
async def current_user(user: str = Depends(get_current_user)):
    return "Hello {}, are you admin ? {}".format(user['username'], user['is_admin'])


@api.get('/predictions/', tags=['predictions'])
async def get_predict(file: str = "https://images.mushroomobserver.org/320/1536252.jpg", 
                      nb_preds: int=1, 
                      user: str = Depends(get_current_user)):
    try:
        
        filename = urlparse(file)
        filename = os.path.basename(filename.path)

        pred_result = Database.is_already_predicted(filename)

        if pred_result is None:
            pred_result = get_predictions(file, nb_preds)
            chpy_db.save_prediction(user['username'], file, pred_result)
        else:
            print("Prediction found")

        pred_json = json.dumps(pred_result)
        resp = Response(pred_json, media_type="application/json")
        
        return resp
    
    except ValueError:
        raise HTTPException(
            status_code=418,
            detail='The file must be an image')
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail='file not found')
    except TypeError:
        raise HTTPException(
            status_code=418,
            detail='nb_preds must be an integer')
    except DbConnexError:
        raise HTTPException(
            status_code=512,
            detail='Database connection failed')
    

@api.get('/accuracy/', tags=['supervise'])
async def get_model_accuracy(user: str = Depends(get_current_user)):
    
    if not user['is_admin']:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="admin access required",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    acc_result = get_accuracy()

    acc_json = json.dumps(acc_result)
    resp = Response(acc_json, media_type="application/json")
    
    return resp

@api.get('/past_pred_acc/', tags=['supervise'])
async def get_last_predictions_accuracy(nb_last_preds:int = 10, 
                                        user: str = Depends(get_current_user)):
    
    if not user['is_admin']:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="admin access required",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    acc_result = chpy_db.get_last_predictions_accuracy(nb_last_preds)

    acc_json = json.dumps(acc_result)
    resp = Response(acc_json, media_type="application/json")
    
    return resp

@api.get('/nb_new_img/', tags=['supervise'])
async def get_nb_new_images(model_name: str="VGG16", 
                            stage: str = "Production",
                            user: str = Depends(get_current_user)):
    
    if not user['is_admin']:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="admin access required",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    model_date = get_model_date(model_name, stage)
    nb_images = chpy_db.get_nb_images_since(model_date)
    
    return nb_images

@api.get('/finetune/', tags=['supervise'])
async def fine_tune_model(model_name: str="VGG16", 
                          stage: str = "Production", 
                          variables:int = 2, 
                          epochs:int = 10, 
                          user: str = Depends(get_current_user)):
    
    if not user['is_admin']:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="admin access required",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(root_dir, 'predict', 'images_temp')

    ft.download_images_for_dataset(images_dir)
    train_ds, val_ds, nb_classes = ft.create_datasets(images_dir) 
    model = ft.load_model(model_name, stage)

    model = ft.fine_tune_model(model, variables)
    model = ft.compile_model(model)
    history = ft.train_model(model, epochs, train_ds, val_ds)

    result = ft.get_history_last_values(history)

    res_json = json.dumps(result)
    resp = Response(res_json, media_type="application/json")
    
    return resp

