from fastapi import FastAPI, Request, HTTPException
import logging
import pandas as pd
import json
import base64
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os
from fuzzywuzzy import fuzz, process
import asyncio
from pydantic import BaseModel
import re
import numpy as np
from collections import defaultdict
import time
from functools import lru_cache

# --------------------------
# Configuration
# --------------------------
OPENAI_PROXY_URL = os.getenv("OPENAI_PROXY_URL", "https://turbo.torob.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "trb-2c4a887cfff351fca8-8183-465b-afd3-07ab64937bb3")
DATA_DIR = os.getenv("DATA_DIR", "./data")
MAX_CONVERSATION_TURNS = 5
RESPONSE_TIMEOUT = 10.0

# --------------------------
# Logging setup
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Pydantic Models
# --------------------------
class Message(BaseModel):
    role: str
    content: str
    type: Optional[str] = "text"

class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None

class ConversationState(BaseModel):
    history: List[Dict] = []
    context: Dict = {}
    turn_count: int = 0
    current_intent: Optional[str] = None
    potential_products: List[str] = []
    clarification_stage: Optional[str] = None
    last_activity: datetime = datetime.now()

# --------------------------
# In-memory conversation storage with cleanup
# --------------------------
conversation_states: Dict[str, ConversationState] = {}

def cleanup_old_conversations():
    """Remove conversations older than 1 hour"""
    cutoff = datetime.now() - timedelta(hours=1)
    to_remove = [
        chat_id for chat_id, state in conversation_states.items()
        if state.last_activity < cutoff
    ]
    for chat_id in to_remove:
        del conversation_states[chat_id]
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} old conversations")

# --------------------------
# Load and preprocess datasets
# --------------------------
tables = ["searches", "base_views", "final_clicks", "base_products", 
          "members", "shops", "categories", "brands", "cities"]

data = {}
search_indices = {}

for table in tables:
    try:
        file_path = os.path.join(DATA_DIR, f"{table}.parquet")
        data[table] = pd.read_parquet(file_path)
        logger.info(f"Loaded {file_path} with {len(data[table])} rows")
    except Exception as e:
        logger.error(f"Failed to load {table}.parquet from {DATA_DIR}: {e}")
        data[table] = pd.DataFrame()

# Create search indices for better performance
def create_search_indices():
    """Create optimized search structures"""
    global search_indices
    
    if not data["base_products"].empty:
        # Product name search index
        products = data["base_products"].copy()
        products["search_text"] = (
            products["persian_name"].astype(str) + " " + 
            products["english_name"].astype(str)
        ).str.lower()
        
        # Extract product codes from names
        products["codes"] = products["search_text"].apply(extract_product_codes)
        
        search_indices["products"] = products
        search_indices["product_codes"] = {}
        
        for idx, row in products.iterrows():
            for code in row["codes"]:
                if code:
                    search_indices["product_codes"][code.lower()] = row["random_key"]
    
    # Category hierarchy
    if not data["categories"].empty:
        search_indices["categories"] = build_category_hierarchy()
    
    # Price index for quick lookups
    if not data["members"].empty:
        price_index = defaultdict(list)
        for idx, row in data["members"].iterrows():
            price_index[row["base_random_key"]].append({
                "member_key": row["random_key"],
                "shop_id": row["shop_id"],
                "price": row["price"]
            })
        search_indices["prices"] = dict(price_index)

def extract_product_codes(text: str) -> List[str]:
    """Extract product codes from text"""
    codes = []
    # Common patterns: "کد X", "code X", "model X", numbers/letters
    patterns = [
        r'کد\s*([A-Za-z0-9]+)',
        r'code\s*([A-Za-z0-9]+)',
        r'model\s*([A-Za-z0-9]+)',
        r'\b([A-Z]\d+|\d+[A-Z]|[A-Z]{2,}\d+|\d+[A-Z]{2,})\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        codes.extend(matches)
    
    return list(set(codes))

def build_category_hierarchy() -> Dict:
    """Build category hierarchy for better search"""
    categories = data["categories"]
    hierarchy = {}
    
    for idx, row in categories.iterrows():
        cat_id = row["id"]
        parent_id = row["parent_id"]
        title = row["title"]
        
        hierarchy[cat_id] = {
            "title": title,
            "parent_id": parent_id,
            "children": []
        }
    
    # Build parent-child relationships
    for cat_id, info in hierarchy.items():
        parent_id = info["parent_id"]
        if parent_id and parent_id in hierarchy:
            hierarchy[parent_id]["children"].append(cat_id)
    
    return hierarchy

create_search_indices()

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="Torob AI Shopping Assistant", version="2.0")

# --------------------------
# Enhanced OpenAI Client
# --------------------------
class OpenAIClient:
    def __init__(self, proxy_url: str, api_key: str):
        self.proxy_url = proxy_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        self.request_count = 0
        self.response_cache = {}
    
    async def chat_completion(self, messages: List[Dict], model: str = "gpt-4", max_tokens: int = 1000, temperature: float = 0.7):
        """Make a chat completion request with caching"""
        try:
            # Create cache key
            cache_key = f"{model}_{hash(str(messages))}_{max_tokens}_{temperature}"
            
            if cache_key in self.response_cache:
                logger.info("Using cached response")
                return self.response_cache[cache_key]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            self.request_count += 1
            logger.info(f"OpenAI request #{self.request_count}")
            
            response = await self.client.post(
                f"{self.proxy_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Cache successful responses
            self.response_cache[cache_key] = content
            
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

openai_client = OpenAIClient(OPENAI_PROXY_URL, OPENAI_API_KEY)

# --------------------------
# Enhanced Product Search Functions
# --------------------------
@lru_cache(maxsize=1000)
def advanced_product_search(query: str) -> List[Tuple[str, float]]:
    """Advanced product search with multiple strategies"""
    if "products" not in search_indices:
        return []
    
    query_lower = query.lower().strip()
    products_df = search_indices["products"]
    results = []
    
    # Strategy 1: Exact code match (highest priority)
    codes = extract_product_codes(query)
    for code in codes:
        if code.lower() in search_indices["product_codes"]:
            results.append((search_indices["product_codes"][code.lower()], 100.0))
    
    # Strategy 2: Fuzzy matching on names
    if not results:
        choices = []
        for idx, row in products_df.iterrows():
            persian_name = str(row["persian_name"])
            english_name = str(row["english_name"])
            choices.extend([
                (persian_name, row["random_key"]),
                (english_name, row["random_key"])
            ])
        
        matches = process.extractBests(query, 
                                     [choice[0] for choice in choices], 
                                     score_cutoff=60, 
                                     limit=10)
        
        for match in matches:
            # Find corresponding random_key
            for choice in choices:
                if choice[0] == match[0]:
                    results.append((choice[1], match[1]))
                    break
    
    # Strategy 3: Keyword matching in features
    if not results or max(score for _, score in results) < 80:
        for idx, row in products_df.iterrows():
            try:
                features_text = str(row.get("extra_features", "")).lower()
                search_text = str(row.get("search_text", "")).lower()
                combined_text = f"{search_text} {features_text}"
                
                # Check for keyword overlap
                query_words = set(query_lower.split())
                text_words = set(combined_text.split())
                overlap = len(query_words.intersection(text_words))
                
                if overlap > 0:
                    score = (overlap / len(query_words)) * 70
                    if score >= 30:
                        results.append((row["random_key"], score))
            except:
                continue
    
    # Remove duplicates and sort by score
    unique_results = {}
    for key, score in results:
        if key not in unique_results or score > unique_results[key]:
            unique_results[key] = score
    
    return sorted(unique_results.items(), key=lambda x: x[1], reverse=True)

def get_product_info(base_random_key: str) -> Dict:
    """Enhanced product information retrieval"""
    if data["base_products"].empty:
        return {}
    
    df = data["base_products"]
    product = df[df["random_key"] == base_random_key]
    
    if product.empty:
        return {}
    
    product_data = product.iloc[0].to_dict()
    
    # Get category info with hierarchy
    if not data["categories"].empty and product_data.get("category_id"):
        category_info = get_category_path(product_data["category_id"])
        product_data["category_info"] = category_info
    
    # Get brand info
    if not data["brands"].empty and product_data.get("brand_id"):
        brand = data["brands"][data["brands"]["id"] == product_data["brand_id"]]
        if not brand.empty:
            product_data["brand_name"] = brand.iloc[0]["title"]
    
    # Parse extra features
    if product_data.get("extra_features"):
        try:
            product_data["parsed_features"] = json.loads(product_data["extra_features"])
        except:
            product_data["parsed_features"] = {}
    
    return product_data

def get_category_path(category_id: int) -> List[Dict]:
    """Get full category path from root to leaf"""
    if "categories" not in search_indices:
        return []
    
    path = []
    current_id = category_id
    
    while current_id and current_id in search_indices["categories"]:
        cat_info = search_indices["categories"][current_id]
        path.insert(0, {"id": current_id, "title": cat_info["title"]})
        current_id = cat_info["parent_id"]
    
    return path

def get_sellers_info(base_random_key: str) -> List[Dict]:
    """Enhanced seller information with sorting and filtering"""
    if base_random_key not in search_indices.get("prices", {}):
        return []
    
    price_info = search_indices["prices"][base_random_key]
    sellers = []
    
    for item in price_info:
        shop_info = data["shops"][data["shops"]["id"] == item["shop_id"]]
        if not shop_info.empty:
            shop = shop_info.iloc[0]
            city_name = ""
            if not data["cities"].empty:
                city = data["cities"][data["cities"]["id"] == shop["city_id"]]
                if not city.empty:
                    city_name = city.iloc[0]["name"]
            
            sellers.append({
                "member_random_key": item["member_key"],
                "shop_id": item["shop_id"],
                "price": int(item["price"]),
                "shop_score": float(shop["score"]),
                "has_warranty": bool(shop["has_warranty"]),
                "city": city_name
            })
    
    return sorted(sellers, key=lambda x: (x["price"], -x["shop_score"]))

def extract_specific_feature(base_random_key: str, feature_query: str) -> str:
    """Enhanced feature extraction with better matching"""
    product_info = get_product_info(base_random_key)
    if not product_info.get("parsed_features"):
        return "Feature information not available."
    
    features = product_info["parsed_features"]
    query_lower = feature_query.lower()
    
    # Direct key matching
    for key, value in features.items():
        key_lower = key.lower()
        if query_lower in key_lower or key_lower in query_lower:
            return str(value)
    
    # Semantic matching for common queries
    feature_mappings = {
        "عرض": ["width", "عرض", "پهنا"],
        "طول": ["length", "طول", "درازا"],
        "ارتفاع": ["height", "ارتفاع", "بلندی"],
        "وزن": ["weight", "وزن"],
        "رنگ": ["color", "colour", "رنگ"],
        "سایز": ["size", "سایز", "اندازه"],
        "جنس": ["material", "جنس", "متریال"]
    }
    
    for persian_term, variations in feature_mappings.items():
        if any(var in query_lower for var in variations):
            for key, value in features.items():
                if any(var in key.lower() for var in variations):
                    return str(value)
    
    return "Requested feature not found."

def get_similar_products(base_random_key: str, limit: int = 5) -> List[str]:
    """Enhanced similar product finding"""
    product_info = get_product_info(base_random_key)
    if not product_info:
        return []
    
    if data["base_products"].empty:
        return []
    
    df = data["base_products"]
    similar_products = []
    
    # Strategy 1: Same category and brand
    same_category_brand = df[
        (df["category_id"] == product_info.get("category_id")) &
        (df["brand_id"] == product_info.get("brand_id")) &
        (df["random_key"] != base_random_key)
    ]
    similar_products.extend(same_category_brand["random_key"].tolist()[:2])
    
    # Strategy 2: Same category, different brand
    if len(similar_products) < limit:
        same_category = df[
            (df["category_id"] == product_info.get("category_id")) &
            (df["brand_id"] != product_info.get("brand_id")) &
            (df["random_key"] != base_random_key) &
            (~df["random_key"].isin(similar_products))
        ]
        similar_products.extend(same_category["random_key"].tolist()[:limit-len(similar_products)])
    
    # Strategy 3: Parent category if still need more
    if len(similar_products) < limit and "category_info" in product_info:
        category_path = product_info["category_info"]
        if len(category_path) > 1:
            parent_cat_id = category_path[-2]["id"]
            parent_category = df[
                (df["category_id"] == parent_cat_id) &
                (df["random_key"] != base_random_key) &
                (~df["random_key"].isin(similar_products))
            ]
            similar_products.extend(parent_category["random_key"].tolist()[:limit-len(similar_products)])
    
    return similar_products[:limit]

def rank_products(base_keys: List[str], ranking_criteria: str = "price") -> List[str]:
    """Rank products based on various criteria"""
    if not base_keys:
        return []
    
    products_with_scores = []
    
    for key in base_keys:
        product_info = get_product_info(key)
        sellers = get_sellers_info(key)
        
        if not sellers:
            continue
        
        # Calculate ranking score
        min_price = min(seller["price"] for seller in sellers)
        avg_shop_score = sum(seller["shop_score"] for seller in sellers) / len(sellers)
        has_warranty_count = sum(1 for seller in sellers if seller["has_warranty"])
        
        # Composite score (lower price is better, higher shop score is better)
        composite_score = (
            (1000000 - min_price) * 0.4 +  # Price weight
            avg_shop_score * 20000 * 0.3 +  # Shop score weight
            has_warranty_count * 10000 * 0.3  # Warranty weight
        )
        
        products_with_scores.append((key, composite_score))
    
    # Sort by score (descending)
    ranked = sorted(products_with_scores, key=lambda x: x[1], reverse=True)
    return [key for key, score in ranked]

# --------------------------
# Enhanced Intent Classification
# --------------------------
async def classify_intent_and_extract_info(query: str, conversation_history: List[Dict] = None) -> Dict:
    """Enhanced intent classification with information extraction"""
    
    # Quick pattern matching for obvious cases
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["قیمت", "فیمت", "price", "تومان", "ریال"]):
        return {"intent": "seller_info", "confidence": 0.9, "extracted_info": {"focus": "price"}}
    
    if any(word in query_lower for word in ["مقایسه", "compare", "بهتر", "better", "کدام", "which"]):
        return {"intent": "comparison", "confidence": 0.9, "extracted_info": {}}
    
    if any(word in query_lower for word in ["مشابه", "similar", "شبیه", "like"]):
        return {"intent": "similar_products", "confidence": 0.9, "extracted_info": {}}
    
    # Feature-specific patterns
    feature_patterns = ["عرض", "طول", "وزن", "رنگ", "سایز", "جنس", "width", "height", "weight"]
    if any(pattern in query_lower for pattern in feature_patterns):
        return {"intent": "feature_query", "confidence": 0.8, "extracted_info": {"feature_type": "specification"}}
    
    # Use LLM for complex cases
    system_prompt = """You are an AI assistant that classifies user intents for an e-commerce platform. 

Classify the user's query into one of these scenarios:
1. direct_search - User wants to find a specific product
2. feature_query - User asks about specific features of a product
3. seller_info - User asks about sellers, prices, or shops
4. comparison - User wants to compare products
5. similar_products - User wants similar product recommendations
6. general_search - User wants to search for multiple products
7. clarification_needed - Query is too vague and needs clarification

Consider the conversation context and extract relevant information. Respond always in persian (Farsi).
Respond with a JSON object: {"intent": "scenario_name", "confidence": 0.8, "extracted_info": {"key": "value"}}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}"}
    ]
    
    if conversation_history:
        context = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in conversation_history[-3:]])
        messages.insert(-1, {"role": "assistant", "content": f"Previous context: {context}"})
    
    response = await openai_client.chat_completion(messages, max_tokens=200, temperature=0.3)
    
    try:
        result = json.loads(response)
        return result
    except:
        return {"intent": "direct_search", "confidence": 0.5, "extracted_info": {}}

# --------------------------
# Multi-turn Conversation Handler
# --------------------------
async def handle_multiturn_conversation(chat_id: str, query: str, state: ConversationState) -> ChatResponse:
    """Handle multi-turn conversations for ambiguous queries"""
    
    if state.turn_count >= MAX_CONVERSATION_TURNS:
        return ChatResponse(message="I apologize, but I couldn't find the specific product you're looking for. Please try a more specific search.")
    
    # First turn - analyze query and search for potential products
    if state.turn_count == 1:
        search_results = advanced_product_search(query)
        
        if len(search_results) == 1 and search_results[0][1] > 80:
            # High confidence single result
            return ChatResponse(
                base_random_keys=[search_results[0][0]],
                message="Found the product you're looking for!"
            )
        elif len(search_results) > 1:
            # Multiple potential products - start clarification
            state.potential_products = [key for key, score in search_results[:10]]
            state.clarification_stage = "category_selection"
            
            # Group by categories
            categories = {}
            for key in state.potential_products:
                product = get_product_info(key)
                if product.get("category_info"):
                    cat_title = product["category_info"][-1]["title"]
                    if cat_title not in categories:
                        categories[cat_title] = []
                    categories[cat_title].append(key)
            
            if len(categories) > 1:
                cat_list = list(categories.keys())[:5]
                return ChatResponse(
                    message=f"I found products in multiple categories: {', '.join(cat_list)}. Which category are you interested in?"
                )
        
        # Vague query - ask for more details
        state.clarification_stage = "general_clarification"
        return ChatResponse(
            message="I need more details to help you find the right product. Could you tell me more about what you're looking for? For example, what type of product, brand preferences, or specific features?"
        )
    
    # Subsequent turns - process clarification
    if state.clarification_stage == "category_selection":
        # User has specified a category
        selected_products = []
        query_lower = query.lower()
        
        for key in state.potential_products:
            product = get_product_info(key)
            if product.get("category_info"):
                for cat_info in product["category_info"]:
                    if query_lower in cat_info["title"].lower():
                        selected_products.append(key)
                        break
        
        if len(selected_products) == 1:
            return ChatResponse(
                base_random_keys=[selected_products[0]],
                message="Perfect! Here's the product from that category."
            )
        elif len(selected_products) > 1:
            state.potential_products = selected_products
            state.clarification_stage = "brand_selection"
            
            # Group by brands
            brands = set()
            for key in selected_products:
                product = get_product_info(key)
                if product.get("brand_name"):
                    brands.add(product["brand_name"])
            
            if len(brands) > 1:
                brand_list = list(brands)[:5]
                return ChatResponse(
                    message=f"Which brand do you prefer? Available brands: {', '.join(brand_list)}"
                )
    
    elif state.clarification_stage == "brand_selection":
        # User has specified a brand
        selected_products = []
        query_lower = query.lower()
        
        for key in state.potential_products:
            product = get_product_info(key)
            brand_name = product.get("brand_name", "").lower()
            if query_lower in brand_name or brand_name in query_lower:
                selected_products.append(key)
        
        if len(selected_products) == 1:
            return ChatResponse(
                base_random_keys=[selected_products[0]],
                message="Great! Here's the product from your preferred brand."
            )
        elif len(selected_products) > 1:
            # Final attempt - return top result
            return ChatResponse(
                base_random_keys=[selected_products[0]],
                message="Based on your preferences, here's the best match I found."
            )
    
    # If we get here, continue with general search
    search_results = advanced_product_search(query)
    if search_results:
        return ChatResponse(
            base_random_keys=[search_results[0][0]],
            message="Based on your additional information, here's what I found."
        )
    
    return ChatResponse(
        message="I'm still having trouble finding the exact product. Could you provide the product name or model number?"
    )

# --------------------------
# Product Comparison Handler
# --------------------------
async def handle_product_comparison(query: str) -> ChatResponse:
    """Handle product comparison queries"""
    
    # Search for products mentioned in the query
    search_results = advanced_product_search(query)
    
    if len(search_results) < 2:
        return ChatResponse(
            message="I need at least two products to compare. Please specify the products you want to compare."
        )
    
    # Take top 2 products for comparison
    product1_key = search_results[0][0]
    product2_key = search_results[1][0]
    
    product1_info = get_product_info(product1_key)
    product2_info = get_product_info(product2_key)
    
    sellers1 = get_sellers_info(product1_key)
    sellers2 = get_sellers_info(product2_key)
    
    # Prepare comparison data
    comparison_data = {
        "product1": {
            "name": product1_info.get("persian_name", "Product 1"),
            "brand": product1_info.get("brand_name", "Unknown"),
            "min_price": min(s["price"] for s in sellers1) if sellers1 else 0,
            "features": product1_info.get("parsed_features", {})
        },
        "product2": {
            "name": product2_info.get("persian_name", "Product 2"),
            "brand": product2_info.get("brand_name", "Unknown"),
            "min_price": min(s["price"] for s in sellers2) if sellers2 else 0,
            "features": product2_info.get("parsed_features", {})
        }
    }
    
    # Use LLM to generate comparison
    comparison_prompt = f"""Compare these two products and recommend which one is better:

Product 1: {comparison_data['product1']['name']} ({comparison_data['product1']['brand']})
Price: {comparison_data['product1']['min_price']:,} Toman
Features: {json.dumps(comparison_data['product1']['features'], ensure_ascii=False)}

Product 2: {comparison_data['product2']['name']} ({comparison_data['product2']['brand']})
Price: {comparison_data['product2']['min_price']:,} Toman
Features: {json.dumps(comparison_data['product2']['features'], ensure_ascii=False)}

Provide a clear recommendation with reasoning. Be concise and focus on the most important differences."""
    
    messages = [
        {"role": "system", "content": "You are a product comparison expert. Provide clear, helpful comparisons."},
        {"role": "user", "content": comparison_prompt}
    ]
    
    comparison_result = await openai_client.chat_completion(messages, max_tokens=300)
    
    # Determine which product to recommend based on price and features
    if comparison_data['product1']['min_price'] <= comparison_data['product2']['min_price']:
        recommended_key = product1_key
    else:
        recommended_key = product2_key
    
    return ChatResponse(
        message=comparison_result or "Both products have their advantages. Consider your specific needs and budget.",
        base_random_keys=[recommended_key]
    )

# --------------------------
# Image Processing Handler
# --------------------------
async def handle_image_query(image_content: str, task: str = "recognize") -> Tuple[str, Optional[str]]:
    """Enhanced image handling with better product mapping"""
    try:
        if task == "recognize":
            prompt = "What is the main object in this image? Provide a concise, one to three word answer."
        else:  # product mapping
            prompt = """Identify this product for e-commerce search. Provide:
1. Product name/type
2. Brand (if visible)
3. Model/variant (if visible)
4. Key distinguishing features

Be specific and concise."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_content}"
                        }
                    }
                ]
            }
        ]
        
        response = await openai_client.chat_completion(messages, model="gpt-4o", max_tokens=150)
        
        if not response:
            return "Could not process image", None
        
        # For product mapping, try to find matching products
        if task == "product_mapping":
            search_results = advanced_product_search(response)
            if search_results and search_results[0][1] > 50:
                return response, search_results[0][0]
        
        return response, None
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return "Error processing image", None

# --------------------------
# Main Response Generation
# --------------------------
async def generate_response(intent: str, query: str, context: Dict, state: ConversationState) -> ChatResponse:
    """Generate appropriate response based on intent with enhanced logic"""
    
    if intent == "direct_search":
        search_results = advanced_product_search(query)
        
        if not search_results:
            return ChatResponse(message="No products found matching your query. Please try different keywords or be more specific.")
        
        if len(search_results) == 1 or search_results[0][1] > 90:
            # High confidence single result
            base_key = search_results[0][0]
            product_info = get_product_info(base_key)
            return ChatResponse(
                base_random_keys=[base_key],
                message=f"Found: {product_info.get('persian_name', 'Product')}"
            )
        elif len(search_results) > 1 and state.turn_count < MAX_CONVERSATION_TURNS:
            # Multiple results - initiate clarification
            state.current_intent = "clarification_needed"
            state.potential_products = [key for key, score in search_results[:5]]
            return await handle_multiturn_conversation(state.chat_id if hasattr(state, 'chat_id') else 'unknown', query, state)
        else:
            # Return best match
            return ChatResponse(
                base_random_keys=[search_results[0][0]],
                message="Here's the best match I found:"
            )
    
    elif intent == "feature_query":
        search_results = advanced_product_search(query)
        
        if not search_results:
            return ChatResponse(message="Please specify which product you're asking about.")
        
        base_key = search_results[0][0]
        feature_info = extract_specific_feature(base_key, query)
        
        return ChatResponse(
            message=feature_info,
            base_random_keys=[base_key] if feature_info != "Requested feature not found." else None
        )
    
    elif intent == "seller_info":
        search_results = advanced_product_search(query)
        
        if not search_results:
            return ChatResponse(message="Please specify which product you're asking about.")
        
        base_key = search_results[0][0]
        sellers = get_sellers_info(base_key)
        
        if not sellers:
            return ChatResponse(message="No seller information found for this product.")
        
        min_price = min(seller["price"] for seller in sellers)
        
        # Return just the number as required by scenario 3
        if "کمترین قیمت" in query or "minimum price" in query.lower():
            return ChatResponse(message=str(min_price))
        else:
            return ChatResponse(
                message=f"Found {len(sellers)} sellers. Minimum price: {min_price:,} Toman",
                base_random_keys=[base_key],
                member_random_keys=[seller["member_random_key"] for seller in sellers[:3]]
            )
    
    elif intent == "comparison":
        return await handle_product_comparison(query)
    
    elif intent == "similar_products":
        search_results = advanced_product_search(query)
        
        if not search_results:
            return ChatResponse(message="Please specify which product you want similar recommendations for.")
        
        base_key = search_results[0][0]
        similar = get_similar_products(base_key, limit=5)
        
        return ChatResponse(
            base_random_keys=similar,
            message=f"Found {len(similar)} similar products"
        )
    
    elif intent == "general_search":
        search_results = advanced_product_search(query)
        
        if not search_results:
            return ChatResponse(message="No products found. Try different search terms.")
        
        # Rank and return top results
        top_keys = [key for key, score in search_results[:10]]
        ranked_keys = rank_products(top_keys)
        
        return ChatResponse(
            base_random_keys=ranked_keys,
            message=f"Found {len(ranked_keys)} products ranked by relevance and value"
        )
    
    elif intent == "clarification_needed":
        return await handle_multiturn_conversation(state.chat_id if hasattr(state, 'chat_id') else 'unknown', query, state)
    
    else:
        return ChatResponse(
            message="I can help you search for products, compare items, find similar products, or answer questions about features and sellers. What would you like to know?"
        )

# --------------------------
# Main Chat Endpoint
# --------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Enhanced main chat endpoint"""
    start_time = time.time()
    
    try:
        chat_id = request.chat_id
        messages = request.messages
        
        # Cleanup old conversations periodically
        if len(conversation_states) > 100:
            cleanup_old_conversations()
        
        # Log the request
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger.info(f"Chat {chat_id}: Processing {len(messages)} messages")
        
        # Save request for debugging
        with open(f"request_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(request.dict(), f, indent=2, ensure_ascii=False)
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        last_message = messages[-1]
        content = last_message.get("content", "")
        message_type = last_message.get("type", "text")
        
        # Initialize or update conversation state
        if chat_id not in conversation_states:
            conversation_states[chat_id] = ConversationState()
        
        state = conversation_states[chat_id]
        state.turn_count += 1
        state.last_activity = datetime.now()
        state.history.append(last_message)
        
        # Timeout check
        if time.time() - start_time > RESPONSE_TIMEOUT - 1:
            return ChatResponse(message="Request timeout. Please try again with a simpler query.")
        
        # Handle sanity check scenarios (Scenario 0)
        if content == "ping":
            return ChatResponse(message="pong")
        elif content.startswith("return base random key:"):
            key = content.split(":", 1)[1].strip()
            return ChatResponse(base_random_keys=[key])
        elif content.startswith("return member random key:"):
            key = content.split(":", 1)[1].strip()
            return ChatResponse(member_random_keys=[key])
        
        # Handle image messages (Scenarios 6 & 7)
        if message_type == "image":
            try:
                # Scenario 6: Object recognition
                object_name, product_key = await handle_image_query(content, "recognize")
                
                if product_key:
                    # Scenario 7: Image to product mapping
                    return ChatResponse(
                        base_random_keys=[product_key],
                        message=f"Identified: {object_name}"
                    )
                else:
                    # Just object recognition
                    return ChatResponse(message=object_name)
                    
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                return ChatResponse(message="Could not process the image. Please try again.")
        
        # Classify intent and generate response
        intent_info = await classify_intent_and_extract_info(
            content, 
            state.history[-3:] if len(state.history) > 1 else None
        )
        
        intent = intent_info.get("intent", "direct_search")
        logger.info(f"Chat {chat_id}: Intent = {intent}, Turn = {state.turn_count}")
        
        # Handle multi-turn scenarios (Scenario 4)
        if (intent == "clarification_needed" or 
            (state.current_intent == "clarification_needed" and state.turn_count <= MAX_CONVERSATION_TURNS)):
            response = await handle_multiturn_conversation(chat_id, content, state)
        else:
            response = await generate_response(intent, content, state.context, state)
        
        # Update conversation state
        state.current_intent = intent
        state.context["last_intent"] = intent
        state.context["last_response"] = response.dict()
        
        # Limit conversation history
        if len(state.history) > 20:
            state.history = state.history[-20:]
        
        # If we found products, end multi-turn conversation
        if response.base_random_keys or response.member_random_keys:
            state.current_intent = None
            state.clarification_stage = None
        
        logger.info(f"Chat {chat_id}: Response generated in {time.time() - start_time:.2f}s")
        return response
        
    except asyncio.TimeoutError:
        return ChatResponse(message="Request timeout. Please try a simpler query.")
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --------------------------
# Additional Endpoints
# --------------------------
@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "loaded_tables": {table: len(df) for table, df in data.items()},
        "search_indices": list(search_indices.keys()),
        "active_conversations": len(conversation_states),
        "openai_requests": openai_client.request_count,
        "cache_size": len(openai_client.response_cache)
    }

@app.get("/stats")
async def get_stats():
    """Get application statistics"""
    total_products = len(data.get("base_products", pd.DataFrame()))
    total_shops = len(data.get("shops", pd.DataFrame()))
    total_categories = len(data.get("categories", pd.DataFrame()))
    
    return {
        "products": total_products,
        "shops": total_shops,
        "categories": total_categories,
        "conversations": len(conversation_states),
        "uptime": datetime.now().isoformat()
    }

@app.delete("/conversations/{chat_id}")
async def clear_conversation(chat_id: str):
    """Clear a specific conversation"""
    if chat_id in conversation_states:
        del conversation_states[chat_id]
        return {"message": f"Conversation {chat_id} cleared"}
    return {"message": "Conversation not found"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Torob AI Shopping Assistant v2.0",
        "scenarios_supported": [
            "Sanity Check (ping/pong)",
            "Direct Product Search",
            "Feature Queries",
            "Seller Information",
            "Multi-turn Conversations", 
            "Product Comparisons",
            "Object Recognition",
            "Image-to-Product Mapping",
            "Similar Products",
            "Product Ranking"
        ],
        "endpoints": ["/chat", "/health", "/stats"]
    }

# --------------------------
# Startup Event
# --------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Torob AI Shopping Assistant v2.0")
    logger.info(f"Loaded {len(data)} data tables")
    logger.info(f"Created {len(search_indices)} search indices")
    
    # Test OpenAI connection
    try:
        test_response = await openai_client.chat_completion([
            {"role": "user", "content": "Hello"}
        ], max_tokens=10)
        if test_response:
            logger.info("OpenAI connection successful")
        else:
            logger.warning("OpenAI connection test failed")
    except Exception as e:
        logger.error(f"OpenAI connection error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")