import chromadb
from sentence_transformers import SentenceTransformer
import logging


ENCODING = 'latin-1'
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
CHROMA_DB_PATH = "./empathetic_rag_db"
COLLECTION_NAME = "empathetic_dialogues_v2"
# v1 test，v2 train
logger = logging.getLogger(__name__)


class EmpatheticRAG:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME):
        logger.info("Initializing EmpatheticRAG system...")
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model '{model_name}' loaded.")
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' is ready. Total documents: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize EmpatheticRAG: {e}")
            raise

    def query(self, situation_text: str, emotion_filter: str = None, top_k: int = 2) -> list:
        where_clause = {}
        if emotion_filter:
            where_clause = {"emotion_id": str(emotion_filter)}

        try:
            results = self.collection.query(
                query_texts=[situation_text],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            return results.get('metadatas', [[]])[0]
        except Exception as e:
            logger.error(f"An error occurred during query: {e}")
            return []


def intelligent_search_cases(rag_system: EmpatheticRAG, current_prompt: str, current_emotion: str, top_k: int = 2) -> list:
    logger.info(f"Attempting precise search with emotion filter: '{current_emotion}'")
    precise_results = rag_system.query(
        situation_text=current_prompt,
        emotion_filter=current_emotion,
        top_k=top_k
    )

    if len(precise_results) >= 1:
        logger.info(f"Precise search successful. Found {len(precise_results)} results.")
        return precise_results

    logger.warning(
        f"Precise search found only {len(precise_results)} result(s). "
        f"Falling back to semantic-only search."
    )
    semantic_results = rag_system.query(
        situation_text=current_prompt,
        top_k=top_k
    )
    return semantic_results


def format_rag_examples_for_prompt(cases: list) -> str:
    if not cases:
        return "No relevant examples found in the knowledge base."

    formatted_string = "### RELEVANT CASE EXAMPLES FROM KNOWLEDGE BASE\n"
    for i, case in enumerate(cases, 1):
        dialogue = case.get('utterances', 'N/A')
        emotion = case.get('emotion_id', 'N/A')
        gold_response = case.get('gold_response', 'N/A')

        formatted_string += f"\n--- Example {i} ---\n"
        formatted_string += f"SITUATION (Emotion: {emotion}):\n{dialogue}\n"
        formatted_string += f"SUCCESSFUL EMPATHETIC RESPONSE:\n{gold_response}\n"
        formatted_string += "--- End of Example ---\n"

    return formatted_string
