import os
import csv
import sqlite3
import sys
import time

from generate import LLM_MODEL, NO_ANSWER, ask
from ingest import CHUNK_OVERLAP, CHUNK_SIZE, PROJECT_ROOT

QUESTIONS_CSV = PROJECT_ROOT / "tests" / "questions.csv"
DB_PATH = PROJECT_ROOT / "results.db"
TOP_K = int(os.getenv("TOP_K", "3"))


def init_db() -> sqlite3.Connection:
    """Create the results table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_label TEXT, model TEXT, chunk_size INTEGER,
            overlap INTEGER, top_k INTEGER,
            question TEXT, category TEXT, expected_answer TEXT,
            answer TEXT, retrieved_chunks TEXT,
            passed INTEGER, notes TEXT
        )"""
    )
    return conn


def normalize(text: str) -> str:
    """Lowercase and drop commas so '8,000' matches '8000'."""
    return text.lower().replace(",", "")


def grade(answer: str, category: str, must_contain: str) -> tuple[int, str]:
    """Return (1 if passed else 0, short note)."""
    said_unknown = "don't know" in normalize(answer) or "do not know" in normalize(answer)
    if category == "unanswerable":
        return (1, "") if said_unknown else (0, "should have refused")
    if said_unknown:
        return 0, "refused but answer exists"
    missing = [m for m in must_contain.split("|") if m and normalize(m) not in normalize(answer)]
    return (0, f"missing: {missing}") if missing else (1, "")


def run(label: str) -> None:
    conn = init_db()
    with open(QUESTIONS_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    passed_total = 0
    for row in rows:
        result = ask(row["question"], k=TOP_K)
        passed, note = grade(result["answer"], row["category"], row["must_contain"])
        passed_total += passed
        chunks = "\n---\n".join(h["text"] for h in result["hits"])
        conn.execute(
            "INSERT INTO results (run_label, model, chunk_size, overlap, top_k,"
            " question, category, expected_answer, answer, retrieved_chunks,"
            " passed, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (label, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
             row["question"], row["category"], row["expected_answer"],
             result["answer"], chunks, passed, note),
        )
        conn.commit()
        print(f"{'PASS' if passed else 'FAIL'} | {row['category']:<12} | {row['question']}")
        time.sleep(2)  # stay under the free-tier rate limit

    print(f"\n{label}: {passed_total}/{len(rows)} passed")
    conn.close()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "baseline")