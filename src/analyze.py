import sqlite3

from run_tests import DB_PATH

conn = sqlite3.connect(DB_PATH)

print("== Accuracy by run ==")
for row in conn.execute(
    """SELECT run_label, chunk_size, overlap, top_k, model,
              SUM(passed) || '/' || COUNT(*) AS score,
              ROUND(100.0 * SUM(passed) / COUNT(*), 1) AS pct
       FROM results GROUP BY run_label ORDER BY MIN(id)"""
):
    print(row)

print("\n== Failures by category (all runs) ==")
for row in conn.execute(
    """SELECT run_label, category, COUNT(*) AS failures
       FROM results WHERE passed = 0
       GROUP BY run_label, category ORDER BY run_label"""
):
    print(row)

print("\n== Every failure in detail ==")
for row in conn.execute(
    "SELECT run_label, question, answer, notes FROM results WHERE passed = 0"
):
    print(row, "\n")