import os
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# DYNAMIC PATHING (No more FileNotFoundError)
# __file__ is fedrag_project/fedrag/client/embedder.py
# .parent (client) -> .parent (fedrag) -> .parent (fedrag_project)
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Loads a lightweight, standard embedding model."""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def load_and_embed(self, dataset_name):
        """Loads a BEIR dataset and converts the text to vectors."""
        dataset_path = DATASETS_DIR / dataset_name

        print(f"\n--- Processing {dataset_name.upper()} ---")
        print(f"Looking for data exactly at: {dataset_path}")

        # Hard gate to ensure the path is perfect before running
        if not dataset_path.exists():
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find data at {dataset_path}")

        # Load the local text files
        corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")

        # Grab just 5 documents for this instant test
        doc_ids = list(corpus.keys())[:5]
        docs = [corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "") for doc_id in doc_ids]

        print(f"Embedding {len(docs)} documents into vector space...")
        embeddings = self.model.encode(docs, show_progress_bar=True)

        print(f"[SUCCESS] Converted text to vectors! Shape of first vector: {embeddings[0].shape}")
        return embeddings


if __name__ == "__main__":
    print("Initiating Local Embedding Test...")
    embedder = LocalEmbedder()
    # Testing on the virology dummy data
    embedder.load_and_embed("trec-covid")
    print("\nPhase 1 COMPLETE. Ready for Federation.")