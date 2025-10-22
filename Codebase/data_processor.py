# data_processor.py

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema.document import Document
from bs4 import BeautifulSoup
from typing import List
import ast

from config import DATA_CSV_PATH

def load_documents_from_csv() -> List[Document]:
    metadata_columns = [
        'p_id', 'name', 'products', 'price', 'colour', 'brand', 'img', 
        'ratingCount', 'avg_rating', 'p_attributes'
    ]
    loader = CSVLoader(
        file_path=DATA_CSV_PATH,
        source_column='p_id',
        csv_args={'delimiter': ','},
        metadata_columns=metadata_columns,
        content_columns=['description'] 
    )
    print("Loading documents from CSV...")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def clean_and_prepare_for_hybrid_search(documents: List[Document]) -> List[Document]:
    """
    Prepares documents for hybrid search. Since the dataset is only for women,
    department inference is not needed.
    """
    print("Preparing documents for Hybrid Search...")
    prepared_docs = []
    for doc in documents:
        if not doc.page_content or not isinstance(doc.page_content, str):
            continue
        
        soup = BeautifulSoup(doc.page_content, 'lxml')
        cleaned_description = soup.get_text(separator=' ', strip=True).replace('\xa0', ' ')
        semantic_content = f"Name: {doc.metadata.get('name', '')}. Type: {doc.metadata.get('products', '')}. Description: {cleaned_description}"
        
        new_metadata = doc.metadata.copy()
        attributes_str = new_metadata.pop('p_attributes', '{}')
        keyword_content = ""
        try:
            attributes_dict = ast.literal_eval(attributes_str)
            if isinstance(attributes_dict, dict):
                keyword_content = " ".join(f"{key.replace(' ', '_')} {value}" for key, value in attributes_dict.items())
        except (ValueError, SyntaxError):
            pass
        
        new_metadata['attributes_text'] = keyword_content
        
        prepared_doc = Document(
            page_content=semantic_content,
            metadata=new_metadata
        )
        prepared_docs.append(prepared_doc)
            
    print(f"Finished preparing {len(prepared_docs)} documents.")
    return prepared_docs