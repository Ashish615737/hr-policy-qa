import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from ingest import DATA_DIR, PROJECT_ROOT, chunk_documents, load_all_pdfs

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "policies"


def build_index(model: SentenceTransformer) -> Collection:
    """Chunk the PDFs, embed every chunk, and store them in Chroma."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Start fresh each time so re-running never creates duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    collection = client.get_or_create_collection(COLLECTION_NAME)

    chunks = chunk_documents(load_all_pdfs(DATA_DIR))
    texts = [c["text"] for c in chunks]

    # One embedding (a list of 384 numbers) per chunk
    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection


def retrieve(
    collection: Collection, model: SentenceTransformer, question: str, k: int = 3
) -> list[dict]:
    """Return the k chunks closest in meaning to the question."""
    question_embedding = model.encode([question]).tolist()
    result = collection.query(query_embeddings=question_embedding, n_results=k)

    hits: list[dict] = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"text": text, "source": meta["source"], "distance": dist})
    return hits


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    collection = build_index(model)
    print(f"Stored {collection.count()} chunks\n")

    test_questions = [
        "How many days of casual leave do I get?",
        "How much is the home internet allowance?",
        "What is the hotel limit in Mumbai?",
        "What is the maternity leave policy?",  # NOT in the documents
    ]
    for question in test_questions:
        print("=" * 70)
        print("Q:", question)
        for hit in retrieve(collection, model, question, k=2):
            print(f"\n  [{hit['source']}] distance={hit['distance']:.3f}")
            print("  " + hit["text"][:200].replace("\n", " "))