import sqlite3

conn = sqlite3.connect("results.db")
row = conn.execute(
    "SELECT retrieved_chunks FROM results WHERE passed = 0"
).fetchone()
print("late marks chunk retrieved:", "late marks" in row[0].lower())