import json
import logging
import threading
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class HybridCatalogSearcher:
    """
    A search engine for the product catalog that uses hybrid retrieval.

    Initialization (happens once at startup):
      1. Load catalog.json into memory
      2. Convert each catalog item into a text "document"
      3. Build a BM25 index over those documents (keyword search)
      4. Generate embeddings for each document (semantic search)
      5. Store embeddings in ChromaDB (vector database)

    At search time (happens on every tool call):
      1. BM25 search: score each document by keyword relevance
      2. Semantic search: find documents with similar embeddings
      3. RRF fusion: combine the two ranked lists
      4. Return the top-K results as a formatted string
    """

    def __init__(self, catalog_path: str, chroma_path: str):
        """
        Initialize the hybrid searcher.

        Args:
            catalog_path: Path to catalog.json
            chroma_path:  Directory where ChromaDB stores its files on disk
        """
        logger.info("Initializing HybridCatalogSearcher...")

        self.catalog = self._load_catalog(catalog_path)
        logger.info(f"Loaded catalog from {catalog_path}")

        self.documents = self._prepare_documents()
        self.doc_texts = [doc["text"] for doc in self.documents]
        self.doc_ids = [doc["id"] for doc in self.documents]
        logger.info(f"Prepared {len(self.documents)} documents for indexing")

        from rank_bm25 import BM25Okapi

        tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info("BM25 index built successfully")

        logger.info(f"Using OpenAI embeddings model: {settings.EMBEDDING_MODEL}")
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI client initialized for embeddings")

        logger.info(f"Initializing ChromaDB at {chroma_path}")
        import chromadb

        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self._init_or_load_collection()
        logger.info("HybridCatalogSearcher initialization complete!")

    def _load_catalog(self, catalog_path: str) -> dict:
        """Load and parse the catalog.json file."""
        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Catalog file not found at {catalog_path}. "
                "Make sure catalog.json exists in the project root."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _prepare_documents(self) -> list[dict]:
        """
        Convert the catalog JSON into a flat list of searchable text documents.

        Each document is a dict with:
          - "id":       Unique identifier for ChromaDB
          - "text":     The full text representation (what we search over)
          - "metadata": The original catalog data (for structured return)


        """
        docs = []

        for plan in self.catalog.get("plans", []):
            features_text = "; ".join(plan.get("features", []))
            not_included_text = "; ".join(plan.get("not_included", []))

            text = (
                f"Plan name: {plan['name']}. "
                f"Monthly price: {plan['price']}. "
                f"Annual price: {plan.get('annual_price', 'N/A')}. "
                f"Maximum users: {plan.get('max_users', 'N/A')}. "
                f"Storage: {plan.get('storage', 'N/A')}. "
                f"Included features: {features_text}. "
                f"Not included in this plan: {not_included_text}. "
                f"Best for: {plan.get('best_for', '')}. "
                f"Free trial: {plan.get('trial', 'N/A')}."
            )

            docs.append(
                {
                    "id": f"plan_{plan['name'].lower().replace(' ', '_')}",
                    "text": text,
                    "metadata": {
                        "type": "plan",
                        "name": plan["name"],
                        "price": plan["price"],
                    },
                }
            )

        for i, faq in enumerate(self.catalog.get("faqs", [])):
            text = f"Question: {faq['question']} " f"Answer: {faq['answer']}"
            docs.append({"id": f"faq_{i}", "text": text, "metadata": {"type": "faq"}})

        summary = self.catalog.get("comparison_summary", "")
        if summary:
            docs.append(
                {
                    "id": "summary",
                    "text": f"Plan comparison summary: {summary}",
                    "metadata": {"type": "summary"},
                }
            )

        return docs

    def _init_or_load_collection(self):
        """
        Initialize or load the ChromaDB collection of embeddings.


        Returns:
            A ChromaDB collection ready for similarity search.
        """
        collection_name = "catalog_embeddings_openai"

        try:
            existing_collection = self.chroma_client.get_collection(collection_name)

            if existing_collection.count() == len(self.documents):
                logger.info(
                    f"Loaded existing ChromaDB collection '{collection_name}' "
                    f"with {existing_collection.count()} embeddings"
                )
                return existing_collection
            else:
                logger.info(
                    f"Catalog changed (collection has {existing_collection.count()} docs, "
                    f"catalog has {len(self.documents)} docs). Recreating collection..."
                )
                self.chroma_client.delete_collection(collection_name)

        except Exception:
            logger.info(f"Creating new ChromaDB collection '{collection_name}'...")

        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity (best for text)
        )

        logger.info(f"Generating embeddings for {len(self.documents)} documents via OpenAI...")
        response = self.openai_client.embeddings.create(
            input=self.doc_texts,
            model=settings.EMBEDDING_MODEL
        )
        embeddings = [item.embedding for item in response.data]

        collection.add(
            embeddings=embeddings,
            documents=self.doc_texts,
            ids=self.doc_ids,
            metadatas=[doc["metadata"] for doc in self.documents],
        )

        logger.info(
            f"ChromaDB collection created with {len(self.documents)} embeddings"
        )
        return collection

    def search(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Perform hybrid search and return formatted results.

        This is the main method called by the search_catalog tool.

        Steps:
          1. BM25 search: get relevance scores for all documents
          2. Semantic search: find top-K semantically similar documents
          3. RRF fusion: combine the two ranked lists into one final ranking
          4. Format top-K results as a human-readable string for the LLM

        Args:
            query: The search query (user's question or keywords)
            top_k: Number of results to return (defaults to settings.TOP_K_SEARCH_RESULTS)

        Returns:
            Formatted string of the most relevant catalog items.
            The LLM uses this as context to answer the user's question.
        """
        if not query or not query.strip():
            return "No search query provided."

        top_k = top_k or settings.TOP_K_SEARCH_RESULTS
        k_rrf = (
            settings.RRF_K_CONSTANT
        )  # The 'k' constant in RRF formula (typically 60)

        logger.debug(f"Hybrid search for query: '{query[:100]}'")

        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_ranked = sorted(
            [(self.doc_ids[i], score) for i, score in enumerate(bm25_scores)],
            key=lambda x: x[1],
            reverse=True,
        )
        bm25_ranked = [(doc_id, score) for doc_id, score in bm25_ranked if score > 0]

        logger.debug(f"BM25 top results: {[doc_id for doc_id, _ in bm25_ranked[:3]]}")

        response = self.openai_client.embeddings.create(
            input=[query],
            model=settings.EMBEDDING_MODEL
        )
        query_embedding = [response.data[0].embedding]

        n_retrieve = min(len(self.documents), top_k * 2)
        semantic_results = self.collection.query(
            query_embeddings=query_embedding, n_results=n_retrieve
        )

        semantic_ids = semantic_results["ids"][0]  # List of document IDs, best-first

        logger.debug(f"Semantic top results: {semantic_ids[:3]}")

        rrf_scores: dict[str, float] = {}

        for rank, (doc_id, _) in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)

        for rank, doc_id in enumerate(semantic_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)

        final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        top_doc_ids = [doc_id for doc_id, _ in final_ranking[:top_k]]

        logger.debug(f"RRF top results: {top_doc_ids}")

        top_documents = []
        for doc_id in top_doc_ids:
            for doc in self.documents:
                if doc["id"] == doc_id:
                    top_documents.append(doc)
                    break

        if not top_documents:
            return "No relevant catalog information found for this query."

        formatted_parts = ["=== Relevant Catalog Information ==="]
        for i, doc in enumerate(top_documents, 1):
            rrf_score = rrf_scores.get(doc["id"], 0.0)
            doc_type = doc["metadata"].get("type", "info")
            formatted_parts.append(
                f"\n[Result {i} | Type: {doc_type} | Relevance: {rrf_score:.4f}]\n"
                f"{doc['text']}"
            )

        return "\n".join(formatted_parts)

    def get_all_documents_text(self) -> str:
        """
        Returns all catalog documents concatenated.
        Used by the eval service to provide full catalog context for scoring.
        """
        return "\n\n".join(self.doc_texts)


_searcher_instance: Optional[HybridCatalogSearcher] = None
_searcher_lock = threading.Lock()


def get_catalog_searcher() -> HybridCatalogSearcher:
    """
    Returns the singleton HybridCatalogSearcher instance.
    Creates it on first call; returns the cached instance on all subsequent calls.

    Thread-safe via double-checked locking:
      - First check (without lock): fast path for the common case (already created)
      - Second check (with lock): ensures only one thread creates the instance
        if two threads both see _searcher_instance is None simultaneously
    """
    global _searcher_instance

    if _searcher_instance is not None:
        return _searcher_instance

    with _searcher_lock:
        if _searcher_instance is None:
            logger.info("Creating HybridCatalogSearcher singleton (first time)...")
            _searcher_instance = HybridCatalogSearcher(
                catalog_path=settings.CATALOG_PATH, chroma_path=settings.CHROMA_DB_PATH
            )
            logger.info("HybridCatalogSearcher singleton created and cached")

    return _searcher_instance
