import pytest
import os.path

from app.predict.predict import get_predictions


def test_get_predictions():
    """Prédiction à partir d'un fichier physique"""
    response = get_predictions(os.path.join("test", "106026.jpg"), 1)
    assert response[0]['name'] == 'Galerina marginata'

    """Prédiction à partir d'une URL"""
    response = get_predictions("https://images.mushroomobserver.org/320/1552315.jpg", 1)
    assert response[0]['name'] == 'Amanita phalloides'
    
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