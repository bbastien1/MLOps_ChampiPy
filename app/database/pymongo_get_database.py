from pymongo import MongoClient
from passlib.context import CryptContext
from pandas import DataFrame

def get_database():

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb+srv://champipy:CCeD3AyOtqxvw2iJ@cluster0.iul9opn.mongodb.net/champipy_db"
    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)
    db = client.champipy_db
    # Create the database for our example (we will use the same database throughout the tutorial
    return db


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

    collection_user.insert_many([user1, user2, user3, user4])    


def get_users():
    chpy_db = get_database()
    collection_users = chpy_db["users"]
    
    users_details = collection_users.find()
    users_df = DataFrame(users_details)

    return users_df.to_json(orient="records")


def get_user(user:str):
    Database = get_database()
    collection_users = Database["users"]
    user_query = { "username": user }
    user = collection_users.find_one(user_query)
    print('username :', user['username'])
    return user["username"]    

def get_user_pwd(user:str):
    Database = get_database()
    collection_users = Database["users"]
    user_query = { "username": user }
    user = collection_users.find_one(user_query)
    
    return user["hashed_password"]


if __name__ == "__main__":   

    # Get the database
    print("TEST !")
    chpy_db = get_database()
    collection_user = chpy_db["users"]
    print(chpy_db.list_collection_names())
    print("return : " + get_user('alice'))
    print("return : " + get_user_pwd('alice'))
