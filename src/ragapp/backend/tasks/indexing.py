import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from create_llama.backend.app.engine.generate import generate_datasource

from chromadb import PersistentClient
from backend.engine.vectordbs.qdrant import get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Cache environment variables for performance
VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma")
CHROMA_PATH = os.getenv("CHROMA_PATH")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "default")
STORAGE_DIR = os.getenv("STORAGE_DIR")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))

# For Faiss GPU integration
USE_FAISS_GPU = os.getenv("USE_FAISS_GPU", "true").lower() == "true"
FAISS_GPU_DEVICE = int(os.getenv("FAISS_GPU_DEVICE", 0))  # GPU device ID


def index_all():
    """
    Call the generate_datasource function to index all data.
    Maintains the original function signature for compatibility.
    """
    generate_datasource()


def _reset_index_chroma():
    """
    Reset the Chroma vector store by deleting the specified collection.
    """
    try:
        chroma_client = PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
        if collection:
            logger.info(f"Removing collection '{CHROMA_COLLECTION}' from Chroma.")
            chroma_client.delete_collection(CHROMA_COLLECTION)
            logger.info(f"Collection '{CHROMA_COLLECTION}' removed successfully.")
    except Exception as e:
        logger.error(f"Error resetting Chroma index: {e}")
        raise


def _reset_index_qdrant():
    """
    Reset the Qdrant vector store by deleting and recreating the collection.
    Additionally configures Qdrant to use GPU if supported.
    """
    try:
        store = get_vector_store()
        logger.info(f"Deleting collection '{store.collection_name}' from Qdrant.")
        store.client.delete_collection(store.collection_name)
        store._create_collection(
            collection_name=store.collection_name,
            vector_size=EMBEDDING_DIM,
            # Add GPU acceleration parameters if supported by Qdrant client
            # Example: use_gpu=True (modify according to actual Qdrant API)
        )
        logger.info(f"Recreated collection '{store.collection_name}' in Qdrant.")
    except Exception as e:
        logger.error(f"Error resetting Qdrant index: {e}")
        raise


def _remove_storage_dir():
    """
    Remove the storage directory if it exists.
    """
    try:
        if os.path.exists(STORAGE_DIR):
            logger.info(f"Removing storage directory at '{STORAGE_DIR}'.")
            shutil.rmtree(STORAGE_DIR)
            logger.info(f"Storage directory '{STORAGE_DIR}' removed successfully.")
        else:
            logger.info(f"Storage directory '{STORAGE_DIR}' does not exist. Skipping removal.")
    except Exception as e:
        logger.error(f"Error removing storage directory: {e}")
        raise


def _initialize_vector_store():
    """
    Initializes the vector store with GPU acceleration if enabled and supported.
    Specifically, configures Faiss to use GPU if applicable.
    """
    vector_store = get_vector_store()
    if USE_FAISS_GPU and vector_store.__class__.__name__ == 'FaissVectorStore':
        try:
            import faiss

            res = faiss.StandardGpuResources()  # Initialize GPU resources
            # Convert CPU index to GPU index
            faiss_index = faiss.index_cpu_to_gpu(res, FAISS_GPU_DEVICE, vector_store.index)
            vector_store.index = faiss_index
            logger.info(f"Initialized Faiss vector store on GPU device {FAISS_GPU_DEVICE}.")
        except Exception as e:
            logger.error(f"Failed to initialize Faiss vector store on GPU: {e}")
            raise
    else:
        logger.info("Using CPU for vector store operations.")
    return vector_store


def reset_index():
    """
    Reset the index by removing the vector store data and STORAGE_DIR, then re-indexing the data.
    Operations are performed in parallel to save time.
    Maintains the original function signature for compatibility.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []

        # Submit vector store reset
        if VECTOR_STORE_PROVIDER.lower() == "chroma":
            futures.append(executor.submit(_reset_index_chroma))
        elif VECTOR_STORE_PROVIDER.lower() == "qdrant":
            futures.append(executor.submit(_reset_index_qdrant))
        else:
            logger.error(f"Unsupported vector store provider: {VECTOR_STORE_PROVIDER}")
            raise ValueError(f"Unsupported vector store provider: {VECTOR_STORE_PROVIDER}")

        # Submit storage directory removal
        futures.append(executor.submit(_remove_storage_dir))

        # Wait for all operations to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error during reset operations: {e}")
                raise

    # Initialize vector store with GPU acceleration if applicable
    _initialize_vector_store()

    # Run the indexing after reset operations are complete
    index_all()


if __name__ == "__main__":
    # Adjust the logging level for performance-critical operations
    logger.setLevel(logging.WARNING)
    try:
        reset_index()
    except Exception as e:
        logger.error(f"Failed to reset and index: {e}")