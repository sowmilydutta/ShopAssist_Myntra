# embedding_generator.py

import os
import shutil
import pickle
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from config import EMBEDDING_MODEL, DB_DIRECTORY, DB_COLLECTION_NAME, BM25_INDEX_PATH
from data_processor import load_documents_from_csv, clean_and_prepare_for_hybrid_search

def main():
    """
    Main pipeline to create both a vector store and a BM25 index for hybrid search.
    """
    print("--- Starting Hybrid Search Indexing Pipeline ---")

    print("1. Deleting old artifacts...")
    if os.path.exists(DB_DIRECTORY):
        shutil.rmtree(DB_DIRECTORY)
    if os.path.exists(BM25_INDEX_PATH):
        os.remove(BM25_INDEX_PATH)
    
    print("2. Loading and preparing documents...")
    raw_docs = load_documents_from_csv()
    documents = clean_and_prepare_for_hybrid_search(raw_docs)
    
    if not documents:
        print("Error: No documents were loaded. Halting pipeline.")
        return

    print("3. Creating vector store (for semantic search)...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=DB_COLLECTION_NAME,
        persist_directory=DB_DIRECTORY
    )
    print(f"Vector store created with {vector_store._collection.count()} documents.")

    print("4. Creating BM25 index (for keyword search)...")
    keyword_corpus = [doc.metadata.get('attributes_text', '') for doc in documents]
    tokenized_corpus = [doc.split(" ") for doc in keyword_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({'bm25': bm25, 'documents': documents}, f)
    print(f"BM25 index saved to '{BM25_INDEX_PATH}'.")

    print("\n--- Hybrid Indexing Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    main()