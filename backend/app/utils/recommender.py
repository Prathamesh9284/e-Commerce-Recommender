import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
DATABASE_NAME = "ecommerce_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load model and tokenizer globally (load once)
logger.info("Loading sentence-transformers model...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Model loaded successfully")

# Initialize Groq LLM
logger.info("Initializing Groq LLM...")
llm = ChatGroq(
    temperature=0.7,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)
logger.info("Groq LLM initialized successfully")

def get_embeddings(texts):
    """Get embeddings using Hugging Face transformer"""
    try:
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = model(**encoded)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logger.debug(f"Generated embeddings for {len(texts)} texts")
        return embeddings.numpy()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def generate_explanation(user_products, recommended_products):
    """
    Generate human-readable explanation for recommendations using LLM
    
    Args:
        user_products: List of products user recently interacted with
        recommended_products: List of recommended products
    
    Returns:
        str: Human-readable explanation
    """
    logger.info("Generating LLM explanation for recommendations")
    
    # Format user products
    user_items = "\n".join([f"- {p['name']} (Brand: {p['brand']}, Price: ₹{p['price']}, Rating: {p['rating']}/5)" 
                            for p in user_products])
    
    # Format recommended products
    rec_items = "\n".join([f"- {p['name']} (Brand: {p['brand']}, Price: ₹{p['price']}, Rating: {p['rating']}/5, Similarity: {p['similarity_score']:.2f})" 
                            for p in recommended_products])
    
    SYSTEM_PROMPT = """
You are an expert e-commerce recommendation assistant that explains the reasoning behind product recommendations.

Your explanations should be:
- Clear and easy to understand
- Based on actual recommendation algorithm factors
- Friendly and conversational
- Focused on helping users understand why these products match their interests
- Around 150-200 words

Key factors to consider in your explanation:
1. **Semantic Similarity**: How similar the recommended products are to what the user viewed (based on product names and features)
2. **Category Matching**: Products are from the same category as the user's recent interest
3. **Price Range**: Recommendations are within ±12.5% of the average price in the category
4. **Rating Filter**: Only products with ratings above the median are recommended
5. **Combined Scoring**: Products are ranked by a weighted score (50% similarity + 30% rating + 20% price value)

Explain in a friendly, conversational tone why these products are good recommendations based on:
- Common patterns in browsing (category, brand preferences)
- How recommended products match their interests
- Value proposition (price, ratings, features)
- What the similarity and overall scores indicate

Keep it natural and engaging."""

    HUMAN_PROMPT = f"""**User's Recent Activities:**
{user_items}

**Recommended Products:**
{rec_items}"""
    
    # Create messages
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    human_message = HumanMessage(content=HUMAN_PROMPT)
    
    # Get LLM response
    response = llm.invoke([system_message, human_message])
    logger.info("LLM explanation generated successfully")
    
    return response.content

def get_recommendations(user_id=None, limit=5):
    """
    Get product recommendations based on user behavior
    
    Args:
        user_id: User ID (optional, defaults to analyzing all user behavior)
        limit: Number of recommendations to return (default: 5)
        include_explanation: Whether to include LLM-generated explanation (default: True)
    
    Returns:
        dict: Dictionary with recommendations and optional explanation
    """
    logger.info(f"Generating recommendations for user_id: {user_id}, limit: {limit}")
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    logger.debug("Connected to MongoDB")
    
    # Build query for user behavior
    user_query = {'user_id': user_id} if user_id else {}
    
    # Get last 10 user behaviors with sorting and limit in MongoDB
    user_behavior = list(db.user_behavior.find(user_query).sort('timestamp', -1).limit(10))
    logger.info(f"Retrieved {len(user_behavior)} user behavior records")
    
    if len(user_behavior) == 0:
        logger.warning(f"No user behavior found for user_id: {user_id}")
        client.close()
        return {
            "recommendations": [],
            "explanation": ""
        }
    
    # Reverse to get chronological order
    user_behavior.reverse()
    
    # Get the most recent product_id
    recent_pid = user_behavior[-1]['product_id']
    logger.info(f"Most recent product_id: {recent_pid}")
    
    # Get the category of the most recent product
    recent_product = db.products.find_one({'product_id': recent_pid}, {'category': 1})
    if not recent_product:
        logger.error(f"Recent product {recent_pid} not found in database")
        client.close()
        return {
            "recommendations": [],
            "explanation": ""
        }
    
    recent_cat = recent_product['category']
    logger.info(f"Recent category: {recent_cat}")
    
    # Get all products in that category from MongoDB
    cat_products_cursor = db.products.find(
        {'category': recent_cat},
        {'_id': 0, 'product_id': 1, 'name': 1, 'brand': 1, 'category': 1, 
            'price': 1, 'rating': 1, 'features': 1, 'stock': 1}
    )
    cat_products = list(cat_products_cursor)
    logger.info(f"Found {len(cat_products)} products in category: {recent_cat}")
    
    if len(cat_products) == 0:
        logger.warning(f"No products found in category: {recent_cat}")
        client.close()
        return {
            "recommendations": [],
            "explanation": ""
        }
    
    # Convert to DataFrame
    prod_df = pd.DataFrame(cat_products)
    prod_df['price'] = prod_df['price'].astype(float)
    prod_df['rating'] = prod_df['rating'].astype(float)
    
    # Calculate price mean and range
    price_mean = prod_df['price'].mean()
    price_min = price_mean * 0.875  # -12.5%
    price_max = price_mean * 1.125  # +12.5%
    logger.info(f"Price range: ₹{price_min:.2f} - ₹{price_max:.2f} (mean: ₹{price_mean:.2f})")
    
    # Filter products in price range in MongoDB
    price_filtered_cursor = db.products.find(
        {
            'category': recent_cat,
            'price': {'$gte': price_min, '$lte': price_max}
        },
        {'_id': 0}
    )
    range_products = list(price_filtered_cursor)
    logger.info(f"Products in price range: {len(range_products)}")
    
    if len(range_products) == 0:
        logger.warning("No products found in price range")
        client.close()
        return {
            "recommendations": [],
            "explanation": ""
        }
    
    range_df = pd.DataFrame(range_products)
    range_df['price'] = range_df['price'].astype(float)
    range_df['rating'] = range_df['rating'].astype(float)
    
    # Get user's viewed products for embedding
    user_product_ids = [b['product_id'] for b in user_behavior]
    user_products_cursor = db.products.find(
        {'product_id': {'$in': user_product_ids}},
        {'_id': 0}
    )
    user_products_list = list(user_products_cursor)
    user_product_names = [p['name'] for p in user_products_list]
    logger.info(f"Generating embeddings for {len(user_product_names)} user products")
    
    # Encode user products and category products
    user_emb = get_embeddings(user_product_names)
    cat_emb = get_embeddings(list(range_df['name']))
    
    # Calculate average user embedding
    user_avg_emb = user_emb.mean(axis=0).reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_avg_emb, cat_emb)[0]
    range_df['similarity'] = similarities
    logger.debug(f"Calculated similarity scores (mean: {similarities.mean():.4f})")
    
    # Filter by similarity score (top 70%)
    similarity_threshold = range_df['similarity'].quantile(0.30)
    filtered_by_similarity = range_df[range_df['similarity'] >= similarity_threshold].copy()
    logger.info(f"Products after similarity filter (>= {similarity_threshold:.4f}): {len(filtered_by_similarity)}")
    
    if len(filtered_by_similarity) == 0:
        logger.warning("No similar products found after similarity filtering")
        client.close()
        return {
            "recommendations": [],
            "explanation": ""
        }
    
    # Filter by rating (>= median)
    rating_threshold = filtered_by_similarity['rating'].median()
    final_products = filtered_by_similarity[filtered_by_similarity['rating'] >= rating_threshold].copy()
    logger.info(f"Products after rating filter (>= {rating_threshold}): {len(final_products)}")
    
    if len(final_products) == 0:
        logger.warning("No products found after rating filtering")
        client.close()
        return {
            "recommendations": [],
            "explanation": ""
        }
    
    # Normalize price and rating
    price_range = final_products['price'].max() - final_products['price'].min()
    rating_range = final_products['rating'].max() - final_products['rating'].min()
    
    if price_range > 0:
        final_products['price_norm'] = 1 - (final_products['price'] - final_products['price'].min()) / price_range
    else:
        final_products['price_norm'] = 1.0
    
    if rating_range > 0:
        final_products['rating_norm'] = (final_products['rating'] - final_products['rating'].min()) / rating_range
    else:
        final_products['rating_norm'] = 1.0
    
    # Combined score: similarity (50%) + rating (30%) + price (20%)
    final_products['score'] = (
        0.5 * final_products['similarity'] + 
        0.3 * final_products['rating_norm'] + 
        0.2 * final_products['price_norm']
    )
    logger.debug("Calculated combined scores")
    
    # Sort by score
    best_matches = final_products.sort_values('score', ascending=False)
    
    # Convert to list of dictionaries
    recommendations = []
    for _, prod in best_matches.head(limit).iterrows():
        recommendations.append({
            'product_id': prod['product_id'],
            'name': prod['name'],
            'brand': prod['brand'],
            'category': prod['category'],
            'price': float(prod['price']),
            'rating': float(prod['rating']),
            'features': prod['features'],
            'similarity_score': float(prod['similarity']),
            'overall_score': float(prod['score'])
        })
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    
    explanation = generate_explanation(user_products_list, recommendations)
    
    # Save recommendations to database
    if recommendations and user_id:
        recommendation_doc = {
            "recommendations": recommendations,
            "explanation": explanation
        }
        
        db.user_recommendations.delete_many({})
        
        # Insert or update recommendations
        db.user_recommendations.insert_one(
            recommendation_doc
        )
        logger.info("Saved recommendations to database")
    
    # Close MongoDB connection
    client.close()
    logger.debug("MongoDB connection closed")
    
    return {
        "recommendations": recommendations,
        "explanation": explanation
    }
