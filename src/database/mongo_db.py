import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

class MongoDB:
    def __init__(self):
        # Temporarily hardcode the connection string
        self.uri = "mongodb://localhost:27017/"
        self.client = MongoClient(self.uri)
        self.db = self.client.get_database("pcse")

    def get_collection(self, collection_name):
        return self.db.get_collection(collection_name)

    def insert_one(self, collection_name, document):
        collection = self.get_collection(collection_name)
        return collection.insert_one(document)

    def insert_many(self, collection_name, documents):
        collection = self.get_collection(collection_name)
        return collection.insert_many(documents)

    def clear_collection(self, collection_name):
        collection = self.get_collection(collection_name)
        result = collection.delete_many({})
        return result.deleted_count

if __name__ == '__main__':
    # Example usage
    db = MongoDB()
    articles_collection = db.get_collection("articles")
    print(f"Connected to MongoDB. Found {articles_collection.count_documents({})} articles.") 