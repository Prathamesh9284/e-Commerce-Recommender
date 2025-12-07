from fastapi import APIRouter, HTTPException
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv
import logging
from pydantic import BaseModel

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/behavior", tags=["User Behavior"])

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
DATABASE_NAME = "ecommerce_db"
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

class UserBehavior(BaseModel):
    user_id: str
    product_id: str
    action: str  
    timestamp: str

@router.post("/add_behavior")
async def add_behavior(behavior: UserBehavior):
    """
    Add a new user behavior record
    """
    behavior_dict = behavior.model_dump()
    db.user_behavior.insert_one(behavior_dict)

    return {
        "message": "User behavior added successfully",
    }

@router.get("/get_behaviors")
async def get_behaviors():
    """
    Get all user behavior records
    """
    behaviors = []
    for b in db.user_behavior.find({}):
        b["_id"] = str(b["_id"])
        behaviors.append(b)
    
    return {
        "behaviors": behaviors
    }
    
@router.get("/get_behavior/{behavior_id}")
async def get_behavior_by_user(behavior_id: str):
    """
    Get behavior records for a specific user
    """
    behaviors = []
    for b in db.user_behavior.find({"_id": ObjectId(behavior_id)}):
        b["_id"] = str(b["_id"])
        behaviors.append(b)
        
    if not behaviors:
        raise HTTPException(status_code=404, detail="No behaviors found for this user")
    
    return {
        "behaviors": behaviors
    }

@router.put("/update_behavior/{behavior_id}")
async def update_behavior(behavior_id: str, updates: UserBehavior):
    """
    Update an existing user behavior record
    """
    update_data = {k: v for k, v in updates.model_dump().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update fields provided")
    
    db.user_behavior.update_one(
        {"_id": ObjectId(behavior_id)},
        {"$set": update_data}
    )
    
    return {
        "message": "Behavior record updated successfully"
    }

@router.delete("/delete_behavior/{behavior_id}")
async def delete_user_behaviors(behavior_id: str):
    """
    Delete all behavior records for a specific user
    """
    result = db.user_behavior.delete_many({"_id": ObjectId(behavior_id)})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="No behaviors found for this id")
    
    return {
        "message": "Behavior record deleted successfully",
    }
