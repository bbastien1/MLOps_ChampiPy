# Documentation du projet MLOps_ChampiPy

## ChampiPy

ChampiPy est un projet de la formation Data Scientist, le but est de prédire l'espèce d'un champignon via une photo.

Le jeu de données d'entrainement a été réduit à 10 espèces de champignons pour accélérer l'entrainement du modèle : 
<ul>
    <li>Aleuria aurantia</li>
    <li>Amanita muscaria</li>
    <li>Coprinus comatus</li>
    <li>Lycogala epidendrum</li>
    <li>Lycoperdon perlatum</li>
    <li>Mycena haematopus</li>
    <li>Pleurotus pulmonarius</li>
    <li>Polyporus squamosus</li>
    <li>Scleroderma citrinum</li>
    <li>Trametes versicolor</li>
</ul>

## Architecture des dossiers

### **app**
Fichiers de l'API organisés de la manière suivante :
<ul>
    <li>api : corps de l'API</li>
    <li>database : fichiers relatifs à la partie BDD</li>
    <li>predict : modèle, code et fichier de référence pour les prédictions</li>
    <li>requirements.txt : librairies du projet</li>
</ul>

### **docker_image**
Ressources nécessaires à la création du container pour l'API.

<ol>
    <li>Lancez setup.sh (linux) ou setup.cmd (windows)</li>
    <li></li>
</ol>

### **docs**
Fichiers de documentation, dont le cahier des charges

### **test**
Dossier contenant les scripts à utiliser avec PyTest : database_test.py et predict_test.py


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

>```python -m pytest test\database_test.py```

>```python -m pytest test\predict_test.py```
