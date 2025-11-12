# RAG Techniques

This directory contains my practical implementations of various retrieval-augmented generation (RAG) techniques in NLP. It serves as a personal learning repository where I experiment with different approaches to improve information retrieval and question answering systems. It contains a collection of Jupyter notebooks that cover various approaches for building, optimizing, and enhancing RAG systems for different data formats and retrieval challenges.


## Notebooks overview

### **Basic RAG systems**
- **PDF retrieval**: Build a simple RAG pipeline using OpenAI embeddings and FAISS to retrieve content from PDF documents.
- **CSV retrieval**: Extend the same pipeline for structured CSV data.

### **Query enhancement**
- **Query transformation**: Apply techniques like paraphrasing and expansion to improve the relevance and comprehensiveness of retrieved information.
- **HyDE (hypothetical document embeddings)**: Transform brief queries into detailed hypothetical documents to better match longer, complex documents in the vector space.

### **Context and content enrichment**
- **Context enrichment window**: Retrieve relevant document chunks along with their neighboring content for improved context comprehension.
- **Contextual compression**: Use LLMs to extract only the most relevant text snippets, avoiding the noise of large chunks.
- **Semantic chunking**: Split documents into meaningful semantic units instead of fixed-size chunks, preserving information coherence.

### **Indexing**
- **Hierarchical indices**: Improve retrieval from long documents by summarizing and indexing at multiple levels (summaries + detailed chunks), combining breadth and depth in retrieval.

### Ranking
- **Fusion retrieval**: Combination of vector-based semantic search with keyword-based BM25 retrieval to leverage both semantic understanding and exact keyword matching for more robust document retrieval.
- **Reranking methods**: Implementation of two reranking strategies using LLMs and cross-encoder models to improve the relevance and quality of initially retrieved documents.

### Explainability and Transparency
- **Explainable retrieval**: Implementation of a retrieval system that not only finds relevant documents but also generates natural language explanations for why those documents were selected, making the process more transparent and trustworthy.

### **Iterative techniques**
* **RAG system with feedback loop**: Integrate user feedback to dynamically refine document relevance and improve answer quality over time — ideal for adaptive systems like education or support bots.
* **Adaptive RAG system**: Automatically classify query types (e.g., factual, opinion, contextual) and choose the most suitable retrieval strategy using LLMs.
* **Self-RAG system with internal reasoning**: Implements a multi-step, decision-aware RAG pipeline with internal checkpoints: it decides whether to retrieve, filters irrelevant contexts, assesses the factual support of generated responses, and scores their utility before choosing the best answer — a lightweight alternative to agentic frameworks.
* **Corrective RAG**: A fallback-aware RAG pipeline that evaluates the relevance of retrieved documents. If confidence is low, it rewrites the query, performs a web search, refines the results into structured knowledge, and uses that to answer. Useful when internal sources are weak or incomplete.

### **Structured retrieval**
* **Hierarchical retrieval with RAPTOR**: Build a multi-level retrieval system that summarizes and indexes documents across semantic layers. Queries begin at the top of the hierarchy and drill down through relevant summaries to surface the most contextually aligned chunks — ideal for long or complex document collections.
* **Graph-based RAG**: Construct a knowledge graph from documents using LLM-extracted concepts and relationships. This structured graph enables reasoning over interconnected ideas, enriching the retrieval process with path-based context discovery and concept explainability — ideal for domains that benefit from semantic structure.

### **Evaluation**
* **Evaluating RAG systems with DeepEval**: Use the `deepeval` library to assess RAG performance on correctness, contextual faithfulness, and relevance — moving beyond surface-level accuracy.


## Acknowledgements
This work is inspired by and builds upon techniques demonstrated in the excellent [RAG Techniques Repository](https://github.com/NirDiamant/RAG_Techniques) of Nir Diamant. I have implemented these techniques as part of my learning journey to deepen my understanding of RAG systems and organize the concepts in a structured way.

## Purpose
This folder serves as both a personal reference and practice ground for RAG techniques that I have studied and implemented. Some techniques are direct applications from the original guide, while others include my own experimentations and modifications.