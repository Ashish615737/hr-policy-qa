from sentence_transformers import SentenceTransformer

from store import MODEL_NAME, build_index, retrieve

# (question, a phrase that must appear in a retrieved chunk to be useful)
CHECKS: list[tuple[str, str]] = [
    ("How many days of casual leave do I get?", "8 days of Casual Leave"),
    ("How much is the home internet allowance?", "Rs 1,000"),
    ("What is the hotel limit in Mumbai?", "Rs 6,000"),
    ("How many earned leave days can be carried forward?", "10 unused days"),
    ("How many late marks cause a deduction?", "Three late marks"),
]


def run_checks(k: int = 3) -> None:
    model = SentenceTransformer(MODEL_NAME)
    collection = build_index(model)

    for question, expected in CHECKS:
        hits = retrieve(collection, model, question, k=k)
        # Rank (1 = best) of the first chunk containing the expected phrase
        rank = next(
            (i + 1 for i, h in enumerate(hits) if expected in h["text"]), None
        )
        status = f"found at rank {rank}" if rank else "MISSING"
        print(f"{status:<18} | {question}")


if __name__ == "__main__":
    run_checks(k=3)