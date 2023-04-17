# Documentation du projet MLOps_ChampiPy

## Overview

Repository du projet ChampiPy dans le cadre de la formation MLOps

## Usage

La fonction **get_predictions** accepte 2 paramètres :
* **upload_file** : Chemin de l'image, HTML ou UNC
* **nb_preds** : Nombre de prédictions à retourner. Valeur par défaut = 1

### Exemple

Retourne une prédiction contenant le nom de l'espèce ainsi que la probabilité associée.

>```python -c "from predict.predict import get_predictions;print(get_predictions('https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/2011-11-15_Aleuria_aurantia_crop.jpg/290px-2011-11-15_Aleuria_aurantia_crop.jpg',1))"```

Lancement de l'API :

>```uvicorn api.main:api --reload```

[Lien vers la documentation de l'API](http://localhost:8000/docs)
## Installation

## Test
Lancement de pytest :

>```python -m pytest test\predict_test.py```