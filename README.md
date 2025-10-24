# 🛍️ Myntra Hybrid RAG Fashion Search Engine

**Intelligent, Generative Recommendation System with LLM Re-ranking**

This project implements an advanced, multi-stage Retrieval-Augmented Generation (RAG) architecture designed to transform traditional e-commerce search into a highly accurate and conversational fashion recommendation experience.

By combining the strengths of semantic understanding, keyword matching, and cognitive LLM filtering, the system provides curated product recommendations and context-aware summaries.

---

## ✨ Key Features

*   **Hybrid Retrieval:** Combines contextual **Semantic Search** (using BGE embeddings in ChromaDB) and attribute-specific **Keyword Search** (using BM25Okapi).
*   **Reciprocal Rank Fusion (RRF):** Intelligently fuses the results from the semantic and keyword retrievers, ensuring high-quality initial candidates.
*   **LLM Re-ranking Layer:** Utilizes a specialized LLM (`GPT-4-Turbo-Preview`) to act as an "Expert Fashion Curator," meticulously re-ranking candidates to select the top 3 most relevant items based on subtle query nuances.
*   **Dual Generative Output:** Provides a rich, multi-faceted response:
    *   **Fashion Assistant Summary:** A stylish, conversational response with tips and product references.
    *   **AI Judge Evaluation:** A transparent, objective relevance score (1-10) and justification for the final recommendations.
*   **Data Optimization:** Implements a specialized preprocessing pipeline to clean HTML from descriptions and flatten structured product attributes for dual-search indexing.

## 📐 System Architecture

The system is engineered in two main stages: Indexing and Retrieval (RAG), orchestrated primarily by **LangChain**.

### 1. Indexing Pipeline (`data_processor.py`, `embedding_generator.py`)

The goal of the indexing stage is to optimize data for the dual retrieval system:

| Search Component | Content Field Created | Source Data Used | Purpose |
| :--- | :--- | :--- | :--- |
| **Vector Store (ChromaDB)** | **Semantic Content** (Cleaned Description, Name, Category) | `description` (HTML stripped) | Understand user intent and context (e.g., "a dress for a summer party"). |
| **BM25 Index** | **Keyword Content** (Flattened Attributes) | `p_attributes` (e.g., "Fabric_Cotton") | Match specific, factual constraints (e.g., "cotton shirt"). |

### 2. Retrieval-Augmented Generation (RAG) Pipeline (`search_engine.py`)

The `FashionSearchEngine` orchestrates the multi-step RAG chain:

1.  **Hybrid Retrieval:** Query is executed against both the Chroma vector store and the BM25 index.
2.  **RRF Fusion:** The ranks from both result sets are merged using Reciprocal Rank Fusion, yielding a list of initial candidates.
3.  **LLM Re-ranking:** The RRF candidates are passed to `GPT-4-Turbo-Preview` (`llm_reranker`), which outputs the indices of the **TOP 3** best matches in a strict JSON format. This acts as a final cognitive filter.
4.  **Generative Output:**
    *   The **Fashion Assistant** (`GPT-4-Turbo`, `temperature=0.7`) creates a friendly, actionable summary.
    *   The **AI Judge** (`GPT-4-Turbo-Preview`, `temperature=0.2`) performs a critical evaluation of the results against the original query.

---

## 🛠️ Setup and Installation

### Prerequisites

*   Python 3.8+
*   **Data:** A CSV file named `Fashion Dataset v2.csv` containing product listings (must include `description`, `p_id`, `name`, and `p_attributes` columns).
*   **API Key:** An OpenAI API key for the generative models.

### 1. Install Dependencies

Install all required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Update the `config.py` file with your specific environment variables:

```python
# config.py

# --- API Key Setup ---
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" 

# --- Data source ---
# Adjust this path to where your CSV data is located
DATA_CSV_PATH = "path/to/Fashion Dataset v2.csv" 
```

### 3. Build the Indexes

The search engine requires the vector store and BM25 index to be pre-built. Run the indexing pipeline once:

```bash
python embedding_generator.py
```

This script will create the following artifacts in your project directory:
*   `vector_db_langchain/` (Chroma Vector Store)
*   `bm25_index.pkl` (BM25 Index)

---

## 🚀 Usage

Once the indexes are built, you can import and use the `FashionSearchEngine` to perform generative searches.

```python
from search_engine import FashionSearchEngine
import pprint

# 1. Initialize the engine (Loads indexes, sets up LLMs and RAG chains)
engine = FashionSearchEngine()

# 2. Run a complex query
query = "I need a royal blue, high-neck casual top made of breathable fabric."

results = engine.search(query)

# 3. Print the Dual Output
print("\n--- Fashion Assistant Summary ---")
pprint.pprint(results['summary'])

print("\n--- AI Judge Evaluation ---")
pprint.pprint(results['evaluation'])

print("\n--- Top 3 Recommended Products (Metadata) ---")
for doc in results['results']:
    print(f"Product ID: {doc.metadata['p_id']}, Name: {doc.metadata['name']}")
```

---

## 💡 Future Enhancements

The project is structured to allow for several exciting future developments:

1.  **Multi-modal Search:** Integrate a **CLIP-based embedding model** to enable searching using product images or a combination of text and images.
2.  **Personalization Layer:** Implement user profiles to track history and feedback, using this data to bias the LLM re-ranking process toward individual style preferences.
3.  **Active Feedback Loop:** Transform the "AI Judge" score into an active learning mechanism. Log the scores and user feedback (e.g., thumbs up/down) to continuously fine-tune the re-ranking models.
4.  **Advanced Query Deconstruction:** Use an initial LLM call to convert complex natural language queries (e.g., "dress for a summer wedding") into structured filters (e.g., `{occasion: "wedding", season: "summer"}`).
