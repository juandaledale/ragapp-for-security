# flake8: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import torch  # For GPU management
from sentence_transformers import SentenceTransformer  # Example embedding model

from app.engine.loaders import get_documents
from app.engine.vectordb import get_vector_store
from app.settings import init_settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core.schema import (
    Document
)



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Signal handling for graceful shutdown
def signal_handler(sig, frame):
    logger.info("Termination signal received. Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)





# Cache configuration values for performance


STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))  # Example batch size
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))  # For parallel processing
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"  # Toggle GPU usage
GPU_DEVICE = int(os.getenv("GPU_DEVICE", 0))  # GPU device ID
VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma")
CHROMA_PATH = os.getenv("CHROMA_PATH", "storage/chromadb")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "default")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))  # GPU device ID
USE_FAISS_GPU = os.getenv("USE_FAISS_GPU", "true")
FAISS_GPU_DEVICE = int(os.getenv("FAISS_GPU_DEVICE", 1536))  # GPU device ID
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2") 

# Initialize device
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
if USE_GPU and DEVICE == "cpu":
    logger.warning("GPU acceleration requested but no GPU found. Falling back to CPU.")

# Load the embedding model with hardware acceleration
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE, trust_remote_code=True)
    logger.info(f"Loaded embedding model '{EMBEDDING_MODEL_NAME}' on device '{DEVICE}'.")
except Exception as e:
    logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
    raise


def get_doc_store():
    """
    Retrieves the document store. If the storage directory exists, it loads from there.
    Otherwise, it initializes a new in-memory document store.
    """
    if os.path.exists(STORAGE_DIR):
        logger.info(f"Loading document store from '{STORAGE_DIR}'.")
        return SimpleDocumentStore.from_persist_dir(STORAGE_DIR)
    else:
        logger.info("Initializing a new in-memory document store.")
        return SimpleDocumentStore()


def run_pipeline(docstore, vector_store, documents):
    """
    Runs the ingestion pipeline to process and embed documents.
    """
    embed_transform =  SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap,
            ),
            embed_transform,  # Use the custom embedding TransformComponent
        ],
        docstore=docstore,
        docstore_strategy="upserts_and_delete",
        vector_store=vector_store,
    )

    # Run the ingestion pipeline and store the results
    nodes = pipeline.run(show_progress=True, documents=documents)

    return nodes


def persist_storage(docstore, vector_store):
    """
    Persists the storage context to the specified storage directory.
    
    Args:
        docstore: The document store instance.
        vector_store: The vector store instance.
    """
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )
    storage_context.persist(STORAGE_DIR)
    logger.info(f"Persisted storage to '{STORAGE_DIR}'.")


def _batch_generator(data: List[Document], batch_size: int):
    """
    Generator that yields batches of data.
    
    Args:
        data (List[Document]): The list of documents to batch.
        batch_size (int): Number of documents per batch.
    
    Yields:
        List[Document]: A batch of documents.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def _process_batch(docstore, vector_store, batch: List[Document], batch_num: int):
    """
    Processes a single batch of documents.
    
    Args:
        docstore: The document store instance.
        vector_store: The vector store instance.
        batch (List[Document]): The batch of documents to process.
        batch_num (int): The batch number.
    """
    logger.debug(f"Processing batch {batch_num} with {len(batch)} documents.")
    try:
        run_pipeline(docstore, vector_store, batch)
        logger.info(f"Completed processing batch {batch_num}.")
    except Exception as e:
        logger.error(f"Error processing batch {batch_num}: {e}")
        raise


def generate_datasource():
    """
    Generates datasource embeddings in batches to optimize performance.
    Maintains the original function signature for compatibility.
    Utilizes GPU acceleration for embedding computations.
    """
    init_settings()
    logger.info("Starting datasource generation.")

    try:
        # Retrieve documents
        documents = get_documents()
        logger.info(f"Retrieved {len(documents)} documents.")

        # Mark documents as public
        for doc in documents:
            doc.metadata["private"] = "false"

        # Retrieve or initialize stores
        docstore = get_doc_store()
        vector_store = get_vector_store()

        # Generate batches
        batches = list(_batch_generator(documents, BATCH_SIZE))
        total_batches = len(batches)
        logger.info(f"Processing {total_batches} batches with batch size {BATCH_SIZE}.")

        # Process batches serially to ensure thread safety with GPU-based embed_model
        if DEVICE == "cuda":
            logger.info("Processing batches serially to ensure thread safety with GPU.")
            for batch_num, batch in enumerate(batches, start=1):
                _process_batch(docstore, vector_store, batch, batch_num)
        else:
            logger.info("Processing batches in parallel using ThreadPoolExecutor.")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {
                    executor.submit(_process_batch, docstore, vector_store, batch, i + 1): i + 1
                    for i, batch in enumerate(batches)
                }

                for future in as_completed(future_to_batch):
                    batch_num = future_to_batch[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_num}: {e}")
                        raise

        # Persist storage after all batches are processed
        persist_storage(docstore, vector_store)

        logger.info("Finished generating the datasource index successfully.")

    except Exception as e:
        logger.error(f"Failed to generate datasource: {e}")
        raise


if __name__ == "__main__":
    generate_datasource()


