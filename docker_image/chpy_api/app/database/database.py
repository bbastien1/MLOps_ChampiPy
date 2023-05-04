from pymongo import MongoClient
from passlib.context import CryptContext
from pandas import DataFrame

class Database:
    """
    """

    DATABASE = None

    def __init__(self):
        CONNECTION_STRING = "mongodb+srv://champipy:CCeD3AyOtqxvw2iJ@cluster0.iul9opn.mongodb.net/champipy_db"
        client = MongoClient(CONNECTION_STRING)
        Database.DATABASE = client.champipy_db

    def create_users():
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        user1 = {
            'username': 'alice',
            'hashed_password': pwd_context.hash('wonderland'),
            'access': ['user']
        }
        user2 =  {
            'username': 'bob',
            'hashed_password': pwd_context.hash('builder'),
            'access': ['user']
        }
        user3 = {
            'username': 'clementine',
            'hashed_password': pwd_context.hash('mandarine'),
            'access': ['user']
        }
        user4 = {
            'username': 'admin',
            'hashed_password': pwd_context.hash('4dm1N'),
            'access': ['user','admin']
        }

        Database.DATABASE["users"].insert_many([user1, user2, user3, user4])    


    def get_user(self, user:str):
        if user :
            collection_users = Database.DATABASE["users"]
            user_query = { "username": user }
            user = collection_users.find_one(user_query)
            return user["username"]    
        else:
            return None

    def get_user_pwd(self, user:str):
        collection_users = Database.DATABASE["users"]
        user_query = { "username": user }
        user = collection_users.find_one(user_query)
        return user["hashed_password"]
    

    def save_prediction(self, user:str, file:str, results):

        prediction = {
            'username': user,
            'file': file,
            'results': results
        }
        
        Database.DATABASE["predictions"].insert_one(prediction)
