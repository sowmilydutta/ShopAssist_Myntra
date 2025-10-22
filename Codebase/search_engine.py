# File: search_engine.py

import pickle
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import BaseRetriever, Document
from rank_bm25 import BM25Okapi
from typing import List

from config import (
    OPENAI_API_KEY, DB_DIRECTORY, DB_COLLECTION_NAME, EMBEDDING_MODEL, JUDGE_MODEL,
    TOP_K_RESULTS, TOP_N_FOR_GENERATIVE, BM25_INDEX_PATH, PRODUCT_CATEGORIES, SUMMARY_MODEL
)

set_llm_cache(SQLiteCache(database_path = "langchain.db"))

def reciprocal_rank_fusion(results: List[List[Document]], k=60) -> List[Document]:
    """ Fuses results from multiple retrievers using RRF. """
    fused_scores = {}
    doc_map = {str(doc.metadata['p_id']): doc for doc_list in results for doc in doc_list}
    
    for doc_list in results:
        for rank, doc in enumerate(doc_list):
            doc_id = str(doc.metadata['p_id'])
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rank + k)

    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_docs = [doc_map[doc_id] for doc_id, _ in reranked_results if doc_id in doc_map]
    return final_docs

class HybridSearchRetriever(BaseRetriever):
    """Custom retriever that combines BM25 and vector search."""
    vector_retriever: BaseRetriever
    bm25_index: BM25Okapi
    documents: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        
        vector_results = self.vector_retriever.get_relevant_documents(query)
            
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_RESULTS]
        bm25_results = [self.documents[i] for i in top_bm25_indices]

        hybrid_results = reciprocal_rank_fusion([vector_results, bm25_results])
        
        print(f"Hybrid search returned {len(hybrid_results)} results for reranking.")
        return hybrid_results[:TOP_K_RESULTS]

class FashionSearchEngine:
    def __init__(self):
        print("Initializing Hybrid Search Engine with Re-ranking...")
        
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
        
        self.llm_summary = ChatOpenAI(model=SUMMARY_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)
        self.llm_judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
        self.llm_reranker = ChatOpenAI(model=JUDGE_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)

        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        vector_store = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embedding_function, collection_name=DB_COLLECTION_NAME)
        
        self.retriever = HybridSearchRetriever(
            vector_retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS}),
            bm25_index=bm25_data['bm25'],
            documents=bm25_data['documents']
        )
        
        self._create_chains()
        print("RAG chains with re-ranking created.")

    def _format_docs_for_rerank(self, docs: List[Document]):
        """Formats documents for the re-ranking prompt, including their original index."""
        return "\n\n".join(
            f"Document Index: {i}\nProduct ID: {doc.metadata.get('p_id')}\nProduct Name: {doc.metadata.get('name')}\nDescription: {doc.page_content}"
            for i, doc in enumerate(docs)
        )

    def _format_docs_for_summary(self, docs: List[Document]):
        """Formats documents for the final summary and evaluation prompts."""
        return "\n\n".join(f"Product: {doc.metadata.get('name')}\nDescription: {doc.page_content}" for doc in docs)

    def _create_chains(self):
        rerank_prompt_template = """
        You are an expert fashion curator. Your task is to re-rank a list of retrieved fashion items based on their relevance to the user's query.
        Analyze the query and each document carefully. Return a JSON object containing the indices of the TOP 3 most relevant documents in descending order of relevance.

        USER QUERY:
        "{question}"

        CANDIDATE DOCUMENTS:
        {context}

        INSTRUCTIONS:
        - Your response MUST be a single, valid JSON object.
        - The JSON object must have a single key: "reranked_indices".
        - The value of "reranked_indices" must be a list of exactly 3 integers, representing the original indices of the best matching documents.
        - Example: If Document 4 is most relevant, followed by Document 1, then Document 7, your response should be:
          {{"reranked_indices": [4, 1, 7]}}

        YOUR JSON RESPONSE:
        """
        rerank_prompt = PromptTemplate.from_template(rerank_prompt_template)
        
        rerank_chain = (
            rerank_prompt 
            | self.llm_reranker 
            | JsonOutputParser()
        )

        def reorder_docs(inputs):
            original_docs = inputs['retrieved_docs']
            reranked_indices = inputs['rerank_result']['reranked_indices']
            final_docs = [original_docs[i] for i in reranked_indices if i < len(original_docs)][:TOP_N_FOR_GENERATIVE]
            return final_docs

        setup_and_retrieval = RunnableParallel(
            retrieved_docs=self.retriever,
            question=RunnablePassthrough()
        )
        
        reranking_and_reordering_chain = RunnablePassthrough.assign(
            rerank_result=lambda x: rerank_chain.invoke({
                "question": x["question"],
                "context": self._format_docs_for_rerank(x["retrieved_docs"])
            })
        ).assign(
            results=reorder_docs 
        )

        summary_prompt = PromptTemplate.from_template(
            """You are a helpful and stylish AI fashion assistant, specialized in providing accurate and inspiring answers to fashion queries.
            You have received a query from a user looking for fashion-related information: "{question}".
            Additionally, you have been given the following context, which contains the top retrieved items from our fashion database:
            <context>
            {context}
            </context>
            Your primary task is to use the information provided in the context to generate a helpful response to the user's query.
            Please adhere to the following guidelines:
            1.  **Be Direct and Relevant:** Directly address the user's specific needs using the information from the top 2-3 search results.
            2.  **Cite Your Sources:** Mention the names of the specific products you are referencing from the context.
            3.  **Be a Stylist:** Offer a brief, stylish tip or suggestion.
            4.  **Be Concise:** Keep your response user-friendly and concise, aiming for 2-4 sentences.
            5.  **Manage Expectations:** Gently remind the user that for details like price, they should check the product pages.
            Your Stylish Summary:"""
        )
        judge_prompt = PromptTemplate.from_template(
            """You are a meticulous and impartial AI system evaluator. Your sole purpose is to judge the quality and relevance of search results based on a user's query.
            You have been given the original user query: "{question}".
            You have also been given the following context:
            <context>
            {context}
            </context>
            Your task is to critically analyze how well the retrieved items in the context match the user's query.
            Please adhere to the following evaluation protocol:
            1.  **Deconstruct the Query:** Identify all key constraints from the user's query.
            2.  **Compare Each Result:** Verify if each item meets every constraint.
            3.  **Assign a Score:** Assign a relevance score from 1 (irrelevant) to 10 (perfect).
            4.  **Justify Your Score:** Provide a brief, objective justification.
            5.  **Strict JSON Output:** Your entire response MUST be a single, valid JSON object with two keys: "score" (integer) and "reason" (string).
            Your JSON Response:"""
        )

        final_processing_chain = RunnableParallel(
            summary=(
                {"context": lambda x: self._format_docs_for_summary(x['results']), "question": lambda x: x['question']}
                | summary_prompt 
                | self.llm_summary 
                | StrOutputParser()
            ),
            evaluation=(
                {"context": lambda x: self._format_docs_for_summary(x['results']), "question": lambda x: x['question']}
                | judge_prompt 
                | self.llm_judge 
                | JsonOutputParser()
            ),
            results=lambda x: x['results'],
            initial=lambda x: x["retrieved_docs"]
        )

        self.full_chain = (
            setup_and_retrieval
            | reranking_and_reordering_chain
            | final_processing_chain
        )

    def _expand_query_with_category(self, query: str) -> str:
        """
        Finds a product category in the query and prepends it to give it more weight.
        """
        query_lower = query.lower()
        for category in PRODUCT_CATEGORIES:
            if f' {category.lower()} ' in f' {query_lower} ':
                expanded_query = f"{category}, {query}"
                return expanded_query
        return query 

    def search(self, query: str):
        """
        Expands the query with category emphasis before invoking the RAG chain.
        """
        if not query or not isinstance(query, str):
            return None
        
        expanded_query = self._expand_query_with_category(query)
        
        print(f"Original Query: '{query}'")
        if expanded_query != query:
            print(f"Expanded Query: '{expanded_query}'")
        
        return self.full_chain.invoke(expanded_query)