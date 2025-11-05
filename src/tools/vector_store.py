"""
Vector store for RAG using Chroma with OpenAI embeddings.

Provides persistent storage and retrieval of document embeddings with metadata filtering.
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import config

logger = logging.getLogger(__name__)


class EngineerRAGStore:
    """Vector store for engineering documents with semantic search capabilities."""

    def __init__(
        self,
        collection_name: str = "engineer_docs",
        persist_directory: str | None = None,
        embedding_model: str | None = None,
    ):
        """Initialize the vector store.

        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the database (default: ./data/chroma_db)
            embedding_model: OpenAI embedding model to use (default: from config.embeddings_model)
        """
        self.collection_name = collection_name

        # Set default persist directory
        if persist_directory is None:
            persist_directory = str(
                Path(__file__).parent.parent.parent / "data" / "chroma_db"
            )

        self.persist_directory = persist_directory

        # Create directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing vector store at: {self.persist_directory}")

        # Use embedding model from config if not specified
        if embedding_model is None:
            embedding_model = config.embeddings_model

        # Initialize OpenAI embeddings
        # Note: API key is read from OPENAI_API_KEY environment variable
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize or load Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        logger.info(f"Vector store '{collection_name}' initialized successfully")

    def add_documents(
        self, documents: list[Document], metadata: dict | None = None
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            metadata: Optional additional metadata to add to all documents

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []

        # Add optional metadata to all documents
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        # Add to vectorstore
        ids = self.vectorstore.add_documents(documents)

        logger.info(f"Added {len(documents)} documents to vector store")
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """Search for similar documents using semantic similarity.

        Args:
            query: Query string to search for
            k: Number of documents to return
            filter_dict: Optional metadata filter (e.g., {"source": "paper.pdf"})
            score_threshold: Optional minimum similarity score (0-1)

        Returns:
            List of most similar documents
        """
        logger.debug(f"Searching for: '{query}' (k={k})")

        if score_threshold is not None:
            # Use similarity search with score threshold
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
            # Filter by score threshold (lower scores are better in Chroma)
            docs = [doc for doc, score in docs_with_scores if score >= score_threshold]
        else:
            # Regular similarity search
            docs = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)

        logger.debug(f"Found {len(docs)} documents")
        return docs

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter_dict: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Search for similar documents and return with similarity scores.

        Args:
            query: Query string to search for
            k: Number of documents to return
            filter_dict: Optional metadata filter

        Returns:
            List of (document, score) tuples
        """
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )

        logger.debug(f"Found {len(docs_with_scores)} documents with scores")
        return docs_with_scores

    def get_retriever(self, search_kwargs: dict | None = None):
        """Get a LangChain retriever interface.

        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 5})

        Returns:
            LangChain retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        self.vectorstore.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")

    def delete_by_metadata(self, filter_dict: dict) -> None:
        """Delete documents matching metadata filter.

        Args:
            filter_dict: Metadata filter (e.g., {"source": "old_paper.pdf"})
        """
        # Get documents matching filter
        docs = self.vectorstore.similarity_search("", k=10000, filter=filter_dict)

        if docs:
            # Extract IDs and delete (filter out None values)
            ids = [
                doc.metadata["id"]
                for doc in docs
                if "id" in doc.metadata and doc.metadata["id"] is not None
            ]
            if ids:
                self.delete_documents(ids)
                logger.info(f"Deleted {len(ids)} documents matching filter")
        else:
            logger.info("No documents found matching filter")

    def clear_collection(self) -> None:
        """Clear all documents from the collection.

        Warning: This will delete all data in the collection!
        """
        logger.warning(f"Clearing collection '{self.collection_name}'")
        self.vectorstore.delete_collection()

        # Reinitialize the collection
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        logger.info("Collection cleared and reinitialized")

    def get_collection_count(self) -> int:
        """Get the total number of documents in the collection.

        Returns:
            Number of documents
        """
        # This is a workaround since Chroma doesn't have a direct count method
        try:
            # Get collection from the client
            collection = self.vectorstore._collection
            return collection.count()
        except Exception:
            logger.exception("Error getting collection count")
            return 0

    def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            List of collection names
        """
        try:
            client = self.vectorstore._client
            collections = client.list_collections()
            return [col.name for col in collections]
        except Exception:
            logger.exception("Error listing collections")
            return []
