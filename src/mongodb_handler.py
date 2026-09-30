"""
MongoDB Integration Module
Handles NoSQL document storage in social_media_db.
Gracefully handles MongoDB connection failures without crashing the application.
"""

import os
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger("MongoDBHandler")

MONGODB_AVAILABLE = False
try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("PyMongo not installed. MongoDB integration will operate in offline mode.")

class MongoDBHandler:
    def __init__(self, uri=None, db_name="social_media_db"):
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = db_name
        self.client = None
        self.db = None
        self.is_connected = False
        
        self.connect()
        
    def connect(self):
        if not MONGODB_AVAILABLE:
            self.is_connected = False
            return False
            
        try:
            # 2 second timeout to prevent blocking Streamlit app on startup
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.is_connected = True
            logger.info(f"Successfully connected to MongoDB at {self.uri}")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            self.is_connected = False
            self.client = None
            self.db = None
            logger.warning(f"MongoDB connection failed: {e}. Running in local dataset mode.")
            return False

    def get_status(self):
        if self.is_connected:
            return {
                'status': 'CONNECTED',
                'message': f'Connected to MongoDB database "{self.db_name}"',
                'badge_color': 'green'
            }
        else:
            return {
                'status': 'NOT CONNECTED',
                'message': 'MongoDB not connected - running in local dataset mode.',
                'badge_color': 'orange'
            }

    def store_processed_posts(self, pandas_df: pd.DataFrame):
        """
        Stores processed posts dataframe into the MongoDB 'posts' collection.
        """
        if not self.is_connected or self.db is None or pandas_df.empty:
            return False, 0
            
        try:
            posts_collection = self.db['posts']
            posts_collection.delete_many({}) # Refresh for new analysis run
            
            records = pandas_df.to_dict(orient='records')
            for r in records:
                r['uploaded_at'] = datetime.now()
                
            result = posts_collection.insert_many(records)
            inserted_count = len(result.inserted_ids)
            logger.info(f"Inserted {inserted_count} posts into MongoDB posts collection.")
            return True, inserted_count
        except Exception as e:
            logger.error(f"Error inserting posts into MongoDB: {e}")
            return False, 0

    def store_analysis_summary(self, summary_data: dict):
        """
        Stores analytical summary in the 'analysis_results' collection.
        """
        if not self.is_connected or self.db is None:
            return False
            
        try:
            results_collection = self.db['analysis_results']
            summary_data['timestamp'] = datetime.now()
            results_collection.insert_one(summary_data)
            logger.info("Saved analysis summary into MongoDB analysis_results collection.")
            return True
        except Exception as e:
            logger.error(f"Error inserting summary into MongoDB: {e}")
            return False
