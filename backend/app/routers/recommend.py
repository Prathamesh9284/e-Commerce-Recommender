from fastapi import APIRouter, UploadFile, File
from utils.recommender import get_recommendations
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import io

load_dotenv()

router = APIRouter(prefix="/api", tags=["Recommendations"])

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
DATABASE_NAME = "ecommerce_db"
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

@router.post("/upload/products")
async def upload_products(
    products: UploadFile = File(...)
):
    """
    Upload and overwrite product catalog data
    """
    # Read products CSV
    products_content = await products.read()
    prod_df = pd.read_csv(io.StringIO(products_content.decode('utf-8')), comment='#')
    
    # Convert to dictionaries
    product_records = prod_df.to_dict('records')
    
    # Overwrite products collection
    db.products.delete_many({})
    db.products.insert_many(product_records)
    
    return {
        "message": "Products uploaded successfully",
    }

@router.post("/upload/user-behavior")
async def upload_user_behavior(
    user_behavior: UploadFile = File(...)
):
    """
    Upload and overwrite user behavior data
    """
    # Read user behavior CSV
    behavior_content = await user_behavior.read()
    user_df = pd.read_csv(io.StringIO(behavior_content.decode('utf-8')))
    
    # Convert to dictionaries
    user_records = user_df.to_dict('records')
    
    # Overwrite user_behavior collection
    db.user_behavior.delete_many({})
    db.user_behavior.insert_many(user_records)

    return {
        "message": "User behavior uploaded successfully",
    }
    
@router.get("/recommendations/{user_id}")
async def get_user_recommendations(user_id: str):
    """
    Generate and store product recommendations for a user with optional LLM explanation
    """
    result = get_recommendations(user_id=user_id, limit=5)
    
    return result
    
@router.get("/stored/recommendations")
async def get_stored_user_recommendations():
    """
    Retrieve stored recommendations for a user from database
    """
    result = list(db.user_recommendations.find(
        {},
        {"_id": 0}
    ))

    if result:
        return result[0]
    else:
        return {
            "recommendations": [],
            "explanation": ""
        }