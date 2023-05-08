import sys
import os
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from app.database.database import Database


@pytest.fixture
def chpy_db():
    '''Returns the MongoDB database for ChampiPy'''
    return Database()


def test_check_db_connex(chpy_db):
    '''Check connection to the database'''
    assert chpy_db.check_db_connex() == True


# def test_check_api_db_connex(chpy_db):
#     '''Check if the API can connect to the database'''
#     r = requests.get('http://127.0.0.1:8000/dbconnex')
#     assert r.text == 'true'