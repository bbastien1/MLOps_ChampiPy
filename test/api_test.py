import sys
import os
import pytest
from httpx import AsyncClient, BasicAuth

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from app.api.main import api

@pytest.mark.anyio
async def test_status():
    async with AsyncClient(app=api, base_url="http://test") as ac:
        response = await ac.get("/status")
    assert response.status_code == 200
    assert response.json() == True


@pytest.mark.anyio
async def test_db_connex():
    async with AsyncClient(app=api, base_url="http://test") as ac:
        response = await ac.get("/dbconnex")
    assert response.status_code == 200
    assert response.json() == True


@pytest.mark.anyio
async def test_prediction():
    async with AsyncClient(app=api, base_url="http://test") as ac:
        auth = BasicAuth("admin", "admin")
        params = {'file': 'https://images.mushroomobserver.org/320/1536252.jpg',
                  'nb_preds': '1'}
        response = await ac.get("/predictions", params=params, auth=auth)
    assert response.status_code == 200
    assert response.json()[0]['name'] == 'Amanita muscaria'    


@pytest.mark.anyio
async def test_accuracy():
    async with AsyncClient(app=api, base_url="http://test") as ac:
        auth = BasicAuth("admin", "admin")
        response = await ac.get("/accuracy", auth=auth)
    assert response.status_code == 200
    assert response.json() >= 70 


@pytest.mark.anyio
async def test_accuracy():
    async with AsyncClient(app=api, base_url="http://test") as ac:
        auth = BasicAuth("admin", "admin")
        response = await ac.get("/nb_new_img", auth=auth)
    assert response.status_code == 200
    assert response.json() < 500 