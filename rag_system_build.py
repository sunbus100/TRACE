import pandas as pd
import pprint
from rag_handler import EmpatheticRAG, ENCODING, COLLECTION_NAME, CHROMA_DB_PATH

DATA_FILE = 'dataset/data_train.csv'
BATCH_SIZE = 200


def build_knowledge_base(rag_system: EmpatheticRAG, data_file: str):
    print(f"\nAttempting to build knowledge base from '{data_file}'...")
    try:
        df = pd.read_csv(data_file, encoding=ENCODING)
        required_cols = ['utterances', 'prompt', 'emotion_id', 'gold_response']
        df.dropna(subset=required_cols, inplace=True)
        for col in required_cols:
            df[col] = df[col].astype(str)

        print(f"Data loaded successfully. Found {len(df)} rows to process.")
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if rag_system.collection.count() >= len(df):
        print("Knowledge base appears to be up-to-date. Skipping build process.")
        return

    print("Building RAG knowledge base (this may take a while for the first time)...")
    total_rows = len(df)
    for i in range(0, total_rows, BATCH_SIZE):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        documents_to_embed = batch_df['prompt'].tolist()
        metadatas = batch_df[required_cols].to_dict('records')
        ids = [str(j) for j in batch_df.index]

        rag_system.collection.add(
            documents=documents_to_embed,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Processed batch {i // BATCH_SIZE + 1}/{(total_rows // BATCH_SIZE) + 1}...")

    print("Knowledge base build complete!")
    print(f"Total documents in collection: {rag_system.collection.count()}")


if __name__ == "__main__":
    # This part is for testing
    rag_system = EmpatheticRAG(
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    build_knowledge_base(rag_system, DATA_FILE)
    print("\n" + "=" * 50)
    print("      TEST QUERY: SEARCH WITH METADATA FILTER")
    print("=" * 50)
    test_query = "I felt guilty when I was driving home one night and a person tried to fly into my lane, and didn't see me. I honked and they swerved back into their lane, slammed on their brakes, and hit the water cones."
    retrieved_cases = rag_system.query(test_query, emotion_filter="sadness")

    print("\n--- Retrieved Cases (filtered for 'sadness'): ---")
    pprint.pprint(retrieved_cases)
