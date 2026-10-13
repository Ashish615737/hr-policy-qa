from fastapi import FastAPI
from pydantic import BaseModel, Field

from generate import ask

app = FastAPI(title="Policy Q&A Assistant")


class Question(BaseModel):
    question: str = Field(min_length=3)


@app.post("/ask")
def ask_endpoint(body: Question) -> dict:
    """Answer a policy question using only the indexed documents."""
    result = ask(body.question)
    sources = [{"file": h["source"], "text": h["text"]} for h in result["hits"]]
    return {"answer": result["answer"], "sources": sources}