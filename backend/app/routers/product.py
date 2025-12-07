from fastapi import APIRouter, HTTPException, Query
from pymongo import MongoClient
from typing import Optional, List
import os
from dotenv import load_dotenv
import logging
from pydantic import BaseModel

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/products", tags=["Products"])

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
DATABASE_NAME = "ecommerce_db"
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Pydantic models for request/response
class Product(BaseModel):
    product_id: Optional[str] = None
    name: str
    category: str
    price: float
    rating: float
    brand: str
    features: str

@router.post("/add_product", status_code=201)
async def add_product(product: Product):
    """
    Add a new product
    """
    existing = db.products.find_one({"product_id": product.product_id})
    if existing:
        raise HTTPException(status_code=400, detail="Product ID already exists")

    product_dict = product.model_dump()
    db.products.insert_one(product_dict)
    
    logger.info(f"Product {product.product_id} created successfully")
    
    return {
        "message": "Product added successfully",
    }
    

@router.get("/get_products")
async def get_products():
    """
    Get all products with optional filters and pagination
    """
    products = list(db.products.find({}, {"_id": 0}))
    
    return {
        "products": products
    }

@router.get("/get_product_id/{product_id}")
async def get_product(product_id: str):
    """
    Get a single product by ID
    """
    product = db.products.find_one({"product_id": product_id}, {"_id": 0})
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    logger.info(f"Product {product_id} retrieved successfully")
    
    return {
        "product": product
    }

@router.put("/update_product/{product_id}")
async def update_product(product_id: str, product_update: Product):
    """
    Update a product by ID
    """
       
    update_dict = {k: v for k, v in product_update.model_dump().items() if v is not None}
    
    if not update_dict:
        raise HTTPException(status_code=400, detail="No update fields provided")
    
    db.products.update_one(
        {"product_id": product_id},
        {"$set": update_dict}
    )
    
    return {
        "message": "Product updated successfully",
    }
    

@router.delete("/delete_product/{product_id}")
async def delete_product(product_id: str):
    """
    Delete a product by ID
    """
    result = db.products.delete_one({"product_id": product_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")

    return {
        "message": "Product deleted successfully",
    }
