from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


class SecureVectorStore:
    def __init__(self, collection_name="fedrag_medical_index", vector_size=384):
        """
        Initializes a local Qdrant Vector Database.
        This ensures we store Pointers and Math, NEVER raw plaintext patient data.
        """
        print(f"[QDRANT] Booting Secure Vector Store: '{collection_name}'...")
        # Using in-memory storage for rapid simulation testing
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Create the strict vector-only collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        print(f"[QDRANT] Collection created successfully. Distance Metric: COSINE.")

    def index_secure_embeddings(self, doc_ids, embeddings):
        """
        Uploads ONLY the mathematical embeddings and the Document IDs.
        No raw text is passed to this function, ensuring zero privacy leakage.
        """
        print(f"[QDRANT] Uploading {len(doc_ids)} secured vectors to the database...")

        points = []
        for idx, (doc_id, vector) in enumerate(zip(doc_ids, embeddings)):
            # We map the string Document ID to an integer payload for Qdrant compatibility,
            # but we STRICTLY omit any text payload.
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={"doc_id": doc_id}  # Pointer only!
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("[QDRANT] Secure indexing complete.")

    def search(self, query_embedding, limit=10):
        """
        Searches the vector space and returns the Top-K Document IDs and their Relevance Scores.
        """
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )

        # Return a clean list of (Document_ID, Score) tuples
        results = [(hit.payload["doc_id"], hit.score) for hit in search_results]
        return results


if __name__ == "__main__":
    # A quick standalone test to verify the Qdrant engine boots correctly
    import numpy as np

    print("Testing Secure Vector Store Architecture...")
    db = SecureVectorStore()

    # Generate dummy data to test the pointer system
    dummy_ids = ["HOSPITAL_A_DOC_001", "HOSPITAL_A_DOC_002"]
    dummy_embeddings = np.random.rand(2, 384).astype(np.float32)

    db.index_secure_embeddings(dummy_ids, dummy_embeddings)

    dummy_query = np.random.rand(384).astype(np.float32)
    top_results = db.search(dummy_query, limit=2)

    print(f"\n[TEST SUCCESS] Simulated Search Results (Pointers Only):")
    for doc_id, score in top_results:
        print(f" -> {doc_id} (Relevance Score: {score:.4f})")