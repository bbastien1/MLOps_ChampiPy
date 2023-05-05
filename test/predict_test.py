import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pytest
from app.predict.predict import get_predictions, get_accuracy


def test_get_predictions():
    """Prédiction à partir d'un fichier physique"""
    response = get_predictions(os.path.join("test", "1494823.jpg"), 1)
    assert response[0]['name'] == 'Pleurotus pulmonarius'

    """Prédiction à partir d'une URL"""
    response = get_predictions("https://images.mushroomobserver.org/320/1536252.jpg", 1)
    assert response[0]['name'] == 'Amanita muscaria'
    
def test_get_predictions_raises_exception_on_non_picture_argument():
    """Exception pour un format de fichier non pris en charge"""
    with pytest.raises(ValueError):
        get_predictions(os.path.join("test", "106026.html"), 1)

def test_get_predictions_raises_exception_on_not_found_argument():
    """Exception pour un fichier non existant"""
    with pytest.raises(FileNotFoundError):
        get_predictions(os.path.join("test", "nothere.jpg"), 1)

def test_get_predictions_raises_exception_on_non_int_argument():
    """Exception pour un nombre de prédictions non entier"""
    with pytest.raises(TypeError):
        get_predictions(os.path.join("test", "106026.jpg"), '4')

def test_get_accuracy():
    response = get_accuracy()
    assert response >= 0.7