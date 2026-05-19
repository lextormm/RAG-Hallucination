# Advanced Techniques in Retrieval-Augmented Generation (RAG)

While basic Retrieval-Augmented Generation (RAG) involves simply encoding a user query, retrieving the top-K similar document chunks from a vector database, and passing them to a Large Language Model (LLM), production-grade systems require significantly more sophisticated techniques to ensure accuracy and relevance.

## Query Transformation and Routing

User queries are often ambiguous, incomplete, or phrased poorly for vector search. Query transformation techniques modify the query before it hits the retrieval engine.

1. **Query Rewriting:** An LLM is used to rephrase the user's query to make it more explicit and search-friendly. For example, "What's the return policy?" might be rewritten as "What is the 30-day return policy and refund process for electronics?"
2. **HyDE (Hypothetical Document Embeddings):** Instead of using the user query to search the vector database directly, the query is first passed to an LLM to generate a *hypothetical* (and potentially factually incorrect) answer. This hypothetical answer is then embedded and used to search the vector database. This works well because a generated answer often shares more semantic similarity with the target documents than the short query does.
3. **Query Routing:** Complex systems often have multiple data stores (e.g., an SQL database for structured data, a vector database for unstructured data, and a graph database for relationships). A routing mechanism evaluates the query and decides which database(s) to search.

## Advanced Chunking Strategies

How documents are split into chunks dramatically impacts retrieval quality.

1. **Semantic Chunking:** Instead of splitting by a fixed character count (e.g., every 500 characters), semantic chunking splits documents based on semantic boundaries, such as paragraphs or sections. This ensures that thoughts are not cut in half.
2. **Parent-Child Document Retrieval (Small-to-Big Retrieval):** During indexing, documents are split into very small "child" chunks (e.g., individual sentences) and larger "parent" chunks (e.g., full paragraphs). The embeddings of the small child chunks are stored in the vector database. During retrieval, the small chunks provide high precision, but instead of passing the small chunk to the LLM, the system retrieves and passes the entire parent chunk to provide maximum context.

## Re-ranking

Standard vector search (like cosine similarity) is incredibly fast but not always the most accurate for determining true semantic relevance, especially for complex questions. Re-ranking solves this by using a two-stage retrieval process.

1. **Stage 1 (Fast Retrieval):** The vector database rapidly retrieves the top N results (e.g., top 50) using standard embedding similarity.
2. **Stage 2 (Cross-Encoder Re-ranking):** A more computationally expensive Cross-Encoder model evaluates the exact relationship between the user query and each of the N retrieved documents. It scores them based on deep semantic relevance and re-sorts the list. Finally, only the top K (e.g., top 5) from this newly sorted list are passed to the LLM.

Cohere's Re-rank API and sentence-transformers Cross-Encoders are widely used for this purpose.

## Graph RAG

Graph RAG merges traditional vector-based RAG with Knowledge Graphs. 

While vector databases are excellent for finding documents that discuss similar topics, they struggle with multi-hop reasoning (e.g., "Which employee works for the company founded by the person who created this product?"). 

In Graph RAG, entities and relationships are extracted from text during the indexing phase to build a knowledge graph. During retrieval, the system searches both the vector database for text similarity and traverses the knowledge graph to extract explicit relationships, feeding both into the LLM context. This provides highly structured, grounded facts alongside unstructured text.
