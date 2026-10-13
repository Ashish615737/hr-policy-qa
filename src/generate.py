import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from store import CHROMA_DIR, COLLECTION_NAME, MODEL_NAME, retrieve
from ingest import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")  # reads GROQ_API_KEY from .env

# Check the current model name in the Groq console if this one is rejected
LLM_MODEL = "openai/gpt-oss-20b"
NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You answer questions about company policies. "
    "Use ONLY the context provided. "
    f"If the context does not contain the answer, reply exactly: {NO_ANSWER} "
    "If the question has several parts, answer each part separately from the "
    "context, and say you don't know only for a part the context does not cover. "
    "Do not use outside knowledge. Keep answers short and include exact numbers."
)

client = Groq(api_key=os.environ["GROQ_API_KEY"])
embed_model = SentenceTransformer(MODEL_NAME)
collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
    COLLECTION_NAME
)


def build_user_prompt(question: str, hits: list[dict]) -> str:
    """Put the retrieved chunks and the question into one prompt."""
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"


def ask(question: str, k: int = 3) -> dict:
    """Retrieve chunks, call the LLM, return answer plus the chunks used."""
    hits = retrieve(collection, embed_model, question, k=k)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, hits)},
        ],
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "hits": hits}


if __name__ == "__main__":
    for q in [
        "How many days of casual leave do I get?",
        "How many earned leave days can be carried forward?",
        "What is the maternity leave policy?",
    ]:
        result = ask(q)
        print("Q:", q)
        print("A:", result["answer"])
        print("Sources:", [h["source"] for h in result["hits"]], "\n")