import sqlite3

conn = sqlite3.connect("results.db")

# Every failed question, from every run
rows = conn.execute(
    "SELECT run_label, retrieved_chunks FROM results WHERE passed = 0"
)

for label, chunks in rows:
    low = chunks.lower()
    has_late_marks = "late marks" in low
    has_casual_count = "8 days of casual" in low
    print(
        f"{label:<9} | late marks chunk: {has_late_marks}"
        f" | casual leave count chunk: {has_casual_count}"
    )

conn.close()