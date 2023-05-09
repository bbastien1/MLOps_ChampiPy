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
    assert response.json() == True


@pytest.mark.anyio
async def test_db_connex():
    async with AsyncClient(app=api, base_url="http://test") as ac:
        response = await ac.get("/dbconnex")
    assert response.json() == True

@pytest.mark.anyio
async def test_accuracy():
    async with AsyncClient(app=api, base_url="http://test") as ac:
        auth = BasicAuth("admin", "admin")
        response = await ac.get("/accuracy", auth=auth)
    assert response.json() >= int(os.environ["target_accuracy"])


@pytest.mark.anyio
async def test_accuracy():
    async with AsyncClient(app=api, base_url="http://test") as ac:
        auth = BasicAuth("admin", "admin")
        response = await ac.get("/nb_new_img", auth=auth)
    assert response.json() < int(os.environ["target_new_images"])