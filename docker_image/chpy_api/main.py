from fastapi import Depends, FastAPI, HTTPException, status, Query, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext

from predict.predict import get_predictions

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

@api.get('/status', tags=['home'])
async def get_api_status():
    '''
    Retourne True si l'API est fonctionnelle
    '''
    return True


@api.get('/predictions/', tags=['predictions'])
async def get_predict(file: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/2011-11-15_Aleuria_aurantia_crop.jpg/290px-2011-11-15_Aleuria_aurantia_crop.jpg", nb_preds: int=1):
    try:
        return Response(get_predictions(file, nb_preds), media_type="application/json")
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