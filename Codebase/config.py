# config.py

import os

# For example: os.environ.get("OPENAI_API_KEY")
OPENAI_API_KEY = "OPENAI_API_KEY"

# --- Model Configuration ---
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
SUMMARY_MODEL = 'gpt-4-turbo'
JUDGE_MODEL = 'gpt-4-turbo-preview'

# --- Vector Database Configuration ---
DB_DIRECTORY = "vector_db_langchain"
DB_COLLECTION_NAME = "fashion_products_langchain" 

# Number of results to fetch from the vector database
TOP_K_RESULTS = 10 
# Number of top results to pass to the generative model
TOP_N_FOR_GENERATIVE = 3

# --- Caching ---
CACHE_SIZE = 100

# --- Data source ---
DATA_CSV_PATH = r"C:\Users\SOWMILY DUTTA\Python_Projects\Myntra_Search\Fashion Dataset v2.csv"

# --- NEW: Hybrid Search Configuration ---
BM25_INDEX_PATH = "bm25_index.pkl"

PRODUCT_CATEGORIES = [
    'kurtis', 'sarees', 'dresses', 'tshirts', 'jeans', 'palazzos', 'kurtas', 'dhotis', 'anarkalis', 'blousons'
    'trousers', 'shirts', 'pants', 'tops', 'lehengas', 'blouses', 'skirts', 'jackets', 'bralettes', 'gowns', 'salwars', 'shararas'
]
