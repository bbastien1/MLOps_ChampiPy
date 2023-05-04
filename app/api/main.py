from fastapi import Depends, FastAPI, HTTPException, status, Query, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pymongo import errors

from predict.predict import get_predictions
from database.database import Database

import json
from pandas import DataFrame


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
    }
]
)


security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
chpy_db = Database()

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    pwd_hashed = pwd_context.hash(credentials.password)
    
    if chpy_db.check_db_connex():
        #if not(chpy_db.get_user(username)) or not(pwd_context.verify(credentials.password, chpy_db.get_user_pwd(username))):
        if not(chpy_db.get_user(username)) or not(pwd_context.verify(credentials.password, chpy_db.get_user_pwd(username))):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username
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
async def current_user(username: str = Depends(get_current_user)):
    return "Hello {}".format(username)


@api.get('/predictions/', tags=['predictions'])
async def get_predict(file: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/2011-11-15_Aleuria_aurantia_crop.jpg/290px-2011-11-15_Aleuria_aurantia_crop.jpg", nb_preds: int=1, username: str = Depends(get_current_user)):
    try:
    
        pred_result = get_predictions(file, nb_preds)

        chpy_db.save_prediction(username, file, pred_result)

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