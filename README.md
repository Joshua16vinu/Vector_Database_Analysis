# Vector Databases

### **1. FAISS**

- **Purpose:** Highly optimized similarity search on large collections of vectors.
- **Storage:**
    - Embeddings are stored **in-memory** as raw float32 arrays in a matrix.
    - Can be persisted to disk, but primarily designed for **fast in-RAM search**.
- **Indexing:**
    - Supports different index types (flat, IVF, HNSW, etc.) to balance speed and accuracy.
    - In your example: `IndexFlatL2` → no compression, exact L2 distance search.
- **k-NN Search:**
    - Computes L2 distance (or cosine if normalized) between query embedding and stored vectors.
    - Efficiently returns the **top-k nearest neighbors**.

**Key:** FAISS is **compute-focused** and extremely fast, but offers minimal database features. No metadata storage by default.


---

### **2. ChromaDB**

- **Purpose:** Vector database designed for LLM/AI applications with **persistent storage and metadata**.
- **Storage:**
    - Embeddings are stored in a **persistent collection**, typically in a disk-backed database.
    - Each embedding links to a **document ID** and optional metadata (e.g., original query text, timestamps, user info).
- **Indexing:**
    - Handles k-NN search internally with optimized vector indexes.
    - Supports updates, deletes, and new vector additions **without rebuilding the whole index**.
- **k-NN Search:**
    - Computes distances between query embedding and stored embeddings, returning top-k matches.
    - Can directly return associated documents or metadata alongside embeddings.

**Key:** ChromaDB is **database-focused**, supporting persistence, metadata, and seamless integration with applications or LLM pipelines.


**ANALYSIS**

1. BASIC

```markdown

queries = [
    "What is the stock price of Apple?",
    "Show me my portfolio performance",
    "When is my next SIP due?",
    "What is the inflation rate in India?",
    "How can I reduce my credit card debt?",
    "List my top 5 performing mutual funds",
    "Predict Tesla stock for next week",
    "What are my tax saving options?",
    "Give me my EPF balance",
    "Compare Nifty and Sensex performance"
]

```

**FAISS vs ChromaDB Query Comparison**

**Test Query:** `best mutual funds for investment`

**Results:**

**FAISS**

1. List my top 5 performing mutual funds
2. Show me my portfolio performance
3. What are my tax saving options
    
    **Time:** 20.694 ms
    

**ChromaDB**

1. List my top 5 performing mutual funds
2. Show me my portfolio performance
3. What are my tax saving options
    
    **Time:** 16.456 ms
    

**Observations:**

- Both FAISS and ChromaDB returned identical top results.
- ChromaDB was slightly faster (~16.5 ms) than FAISS (~20.7 ms).
- For small datasets, both systems are accurate and efficient.

**Conclusion:**

- Both vector stores are suitable for a financial assistant’s query retrieval system.
- ChromaDB shows minor speed advantage for small-scale data.

1. MEDIUM

**Overview:**

This dataset simulates conversations between users and a financial assistant. It's designed for training, testing, or benchmarking chatbots that provide financial advice.

- **Number of Conversations:** 2,000
- **Turns per Conversation:** 2–6
- **Roles:** `user` and `assistant`

**Data Structure:**

Each conversation turn is stored as a JSON object with:

- `role`: `"user"` or `"assistant"`
- `text`: message content

**User Question Templates:**

- What is the current NAV of {fund}?
- Should I continue my SIP in {fund}?
- How is my portfolio performing?
- Give me investment advice for {fund}.
- What is the risk associated with {fund}?
- How much tax will I pay on my {fund} gains?
- Can you suggest some good mutual funds?
- What's the trend for {fund} this month?
- Is it a good time to buy {fund}?
- How has {fund} performed historically?

**Assistant Response Templates:**

- The NAV for {fund} as of today is ₹{:.2f}.
- Based on your long-term goals, continuing the SIP in {fund} seems reasonable.
- {fund} is currently showing a {trend} trend.
- The risk for {fund} is considered {risk}.
- You may pay around ₹{:.2f} tax on your {fund} gains this year.
- Based on historical performance, {fund} could be a suitable choice.
- Your portfolio has gained/lost {:.2f}% over the past month.
- Considering market conditions, investing in {fund} now seems {trend}.
- I recommend diversifying your investment across equities, bonds, and funds.
- The historical returns of {fund} over the last 5 years average {:.2f}% per annum.

**Sample Funds:**

- HDFC Balanced Advantage Fund
- ICICI Prudential Bluechip Fund
- SBI Small Cap Fund
- Axis Long Term Equity Fund
- Nifty 50 ETF

**Usage:**

- Ideal for training LLMs or chatbots for finance
- Fine-tuning, testing, or embedding generation
- Stored as JSONL for easy integration with Python, Pandas, or vector databases

**Generated Dataset Size:** 2,000 conversations (approximately 8,000 turns)

**Code Workflow:**

```python
import json
import random

# Load synthetic dataset
dataset_file = "synthetic_finance_chatbot_dataset.jsonl"
docs = []
with open(dataset_file, "r") as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        docs.append({
            "id": str(i),
            "role": entry["role"],
            "text": entry["text"]
        })

print(f"Total turns/documents: {len(docs)}")

# Prepare lists for embeddings & FAISS/Chroma
doc_texts = [d["text"] for d in docs]
doc_ids = [d["id"] for d in docs]
```

**Results:**

- **Total turns/documents:** 8,071
- **Embedding dimension:** 384
- **Embeddings type:** NumPy array

**Methodology:**

1. **User Queries:**Examples include:
    - "What is the current NAV of HDFC Balanced Advantage Fund?"
    - "Should I continue my SIP in ICICI Prudential Bluechip Fund?"
    - "How is my portfolio performing this month?"
    - "Give me investment advice for SBI Small Cap Fund."
    - "What tax will I pay on my gains?"
2. **Query Embeddings:**
    - Each user query is converted into numeric embeddings using the same model as the dataset.
3. **Baseline Retrieval:**
    - Dot product similarity is computed between document embeddings and query embeddings.
    - The top-5 closest documents per query establish the **baseline**.
4. **Recall@K:**
    - This metric measures the fraction of relevant documents retrieved in the top-k results.
    - Formula:
        
        Recall@k = (Number of relevant documents in top-k) ÷ (Total number of relevant documents)
        
    - Recall@5 is calculated by comparing retrieved results with the baseline.

**Usage:**

- Recall@k benchmarks vector stores like **FAISS** and **ChromaDB**.
- Higher recall@k indicates that more relevant documents appear in the top results, which improves chatbot response accuracy.

This analysis compares FAISS and ChromaDB vector search performance using synthetic finance chatbot data, measuring insertion time, search time, memory usage, and retrieval accuracy.

**Metrics Comparison:**

| System | Insert Time (s) | Search Time (s) | Memory (MB) | Recall@5 |
| --- | --- | --- | --- | --- |
| FAISS | 0.0332 | 0.0055 | 23.34 | 0.04 |
| ChromaDB | 6.6706 | 0.0058 | 109.79 | 0 |

**Observations:**

- **Insertion Time:** FAISS is significantly faster (0.03s) than ChromaDB (6.67s).
- **Search Time:** Both systems perform similarly (approximately 0.005–0.006s per query).
- **Memory Usage:** FAISS is more efficient, using only 23 MB compared to ChromaDB's 110 MB.
- **Recall@5:** FAISS achieves modest document retrieval (0.04), while ChromaDB retrieves none (0).

**Conclusion:**

- FAISS demonstrates superior speed, memory efficiency, and recall effectiveness for this dataset.
- ChromaDB likely requires optimization of batching processes or embedding settings to improve recall.
- For lightweight, accurate financial assistant retrieval tasks, **FAISS is recommended**.

## WHY RECALL WAS THIS MUCH LOW?

1 reason is because of the synthetic dataset

2nd reason is that documents were not context rich and aligned

**Improving Recall with Context-Aware Documents**

After the initial benchmark:

| System | Insert Time (s) | Search Time (s) | Memory (MB) | Recall@5 |
| --- | --- | --- | --- | --- |
| FAISS | 0.0332 | 0.0055 | 23.34 | 0.04 |
| ChromaDB | 6.6706 | 0.0058 | 109.79 | 0 |

the recall@5 values were notably low. To improve retrieval performance:

**Context-Aware Document Construction:**

1. **Iterate through all conversation turns** in the dataset.
2. **Select assistant turns** as primary retrieval documents.
3. **Concatenate previous user turn** (if available) with the assistant response to create context-enriched documents.
4. **Store results:**
    - `context_docs` → IDs of assistant turns
    - `context_texts` → Combined previous user turn + assistant response

**Purpose:**

- Provides **context-aware semantic search**, improving recall for follow-up queries.
- Enhances the relevance of retrieved documents in the chatbot pipeline.
- Prepares the dataset for **FAISS or ChromaDB insertion** while maintaining conversational continuity.

**Improved Retrieval Performance with Context-Aware Documents**

After restructuring the dataset to include the **previous user turn** for each assistant response and refining embeddings and indexing, the retrieval performance improved dramatically:

| System | Insert Time (s) | Search Time (s) | Memory (MB) | Recall@5 |
| --- | --- | --- | --- | --- |
| FAISS | 0.0107 | 0.0024 | 10.24 | 0.92 |
| ChromaDB | 2.9617 | 0.007 | 15.59 | 0 |

**Observations:**

- **FAISS**
    - Recall@5: 0.92 — Context-aware documents significantly improved accuracy.
- **ChromaDB**
    - Recall@5: 0 — Requires further optimization for context-aware retrieval.

**Conclusion:**

- **FAISS** delivers high recall, low memory usage, and rapid retrieval, making it ideal for context-aware financial assistant queries.
- ChromaDB requires additional tuning or alternative indexing strategies to achieve comparable performance.
- Incorporating **previous user turn context** proves highly effective for enhancing recall in retrieval-augmented generation workflows.

## Why did ChromaDB underperform?

- **Exact vs Approximate Search** – FAISS does exact L2 search, while ChromaDB may use approximate or slower search methods by default.
- **Batching Overhead** – ChromaDB inserted embeddings in large batches, adding overhead and reducing indexing efficiency.
- **Memory & Metadata Management** – ChromaDB handles extra metadata and collection management, consuming more memory and slightly slowing retrieval.
- **Small Dataset** – Your dataset (~8k turns) is small; FAISS’s dense matrix operations are extremely efficient in this range.
- **Context Awareness Missing Initially** – The first ChromaDB run used embeddings without context, lowering recall significantly.
- **Index Optimization** – FAISS stores vectors in contiguous arrays optimized for fast similarity search, while ChromaDB has additional abstraction layers.

## CONCLUSION

**Recall that Counts**

- FAISS soared to **Recall@5 = 0.92** once we included context from the previous user turn. Almost every relevant answer was being retrieved!

**Speed Matters**

- FAISS inserted all embeddings in **0.01s** and retrieved answers in **0.0024s** — almost instantaneous.

**Memory Footprint**

- FAISS kept it lean at **10 MB**, while ChromaDB used **15.6 MB**. Every MB counts when you want scalability.

**Synthetic Data Advantage**

- Our dataset is synthetic (~8k turns), so FAISS’s optimized dense vector search really shines here.
- ChromaDB’s extra features (metadata handling, collections) become overhead in small datasets, slowing things down without improving recall.

### For a **Financial Assistant MVP with agentic workflows**:

- **FAISS** is better if you prioritize:
    - High-speed retrieval
    - Large-scale embedding storage
    - Long-term scalability
    - Integration with custom agents, MCP, and multi-step workflows
- **ChromaDB** is fine for:
    - Quick prototyping
    - Easy setup with embeddings and documents
    - Smaller-scale demos

### **FAISS wins** for small-to-medium datasets, context-aware retrieval, and memory efficiency.

### ChromaDB could shine in large-scale, production environments with complex filtering needs — but for our financial assistant , FAISS is the perfect match.