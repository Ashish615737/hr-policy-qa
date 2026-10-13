# Company Policy Q&A Assistant (RAG)

A Retrieval-Augmented Generation (RAG) assistant that answers employee questions using **only** a company's HR policy documents, and says *"I don't know based on the provided documents."* when the answer is not there. It includes a 25-question test suite, SQL-based result analysis, and a FastAPI endpoint.

Built from scratch (no LangChain, no agents, no Docker) to understand each RAG component.

## Problem

Employees waste time searching long HR, leave and travel policy documents. A chatbot that guesses is risky, because a wrong leave or reimbursement answer causes real problems. The assistant must therefore answer from the documents or refuse.

## Tech stack

| Part | Choice |
|---|---|
| Language | Python 3.13 |
| PDF reading | pypdf |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384 dimensions) |
| Vector store | ChromaDB (local, persistent) |
| LLM | Groq API, `openai/gpt-oss-20b`, temperature 0 |
| Test logging | SQLite |
| API | FastAPI + Uvicorn |
| Config | `.env` file (API key never committed) |

## Documents

Three fictional policies for "Nimbus Analytics Pvt Ltd" (converted to PDF), containing specific numbers so answers can be checked:

- `leave_policy.pdf` (casual, sick and earned leave, probation, unpaid leave, approvals)
- `wfh_policy.pdf` (hybrid schedule, core hours, internet allowance, equipment, late marks)
- `travel_expense_policy.pdf` (hotel limits, meal allowance, claims, reimbursement)

Topics deliberately **not** covered, used for the "unanswerable" tests: maternity/paternity leave, health insurance, stock options, notice period, dress code, relocation, international per-diem.

## Architecture

```
INDEXING (run once)
PDFs --> extract text --> clean text --> 500-char chunks (50 overlap)
     --> embed each chunk (all-MiniLM-L6-v2) --> store in ChromaDB

QUERYING (every question)
Question --> embed --> ChromaDB top-k nearest chunks
         --> prompt = system rules + retrieved chunks + question
         --> Groq LLM (temperature 0)
         --> answer + source chunks

TESTING
questions.csv --> run every question through the pipeline
              --> auto-grade --> log to SQLite --> SQL analysis
```

## Project structure

```
policy-qa/
├── data/                       3 policy PDFs
├── src/
│   ├── ingest.py               load PDFs, clean text, chunk
│   ├── store.py                embed chunks, build/query ChromaDB
│   ├── generate.py             prompt + Groq call (the RAG answer)
│   ├── run_tests.py            run test questions, grade, log to SQLite
│   ├── analyze.py              SQL summaries of results
│   └── api.py                  FastAPI endpoint
├── tests/questions.csv         25 test questions
├── requirements.txt
├── .gitignore                  excludes .env, .venv, chroma_db, *.db
└── README.md
```

## How to run

1. Create and activate a virtual environment:
```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
```
2. Create a free key at console.groq.com and put it in a `.env` file in the project root (no quotes, no spaces):
```
   GROQ_API_KEY=your_key_here
```
3. Build the vector index (run once, and again after changing chunk settings):
```powershell
   python src\store.py
```
4. Ask a question from the command line:
```powershell
   python src\generate.py
```
5. Run the test suite (the label names the run in the database):
```powershell
   python src\run_tests.py baseline
   python src\analyze.py
```
6. Start the API and open http://127.0.0.1:8000/docs:
```powershell
   uvicorn --app-dir src api:app --reload
```

Experiment settings can be changed without editing code:
```powershell
$env:TOP_K=5; python src\run_tests.py k5; Remove-Item Env:TOP_K
$env:CHUNK_SIZE=300; python src\store.py; python src\run_tests.py chunk300; Remove-Item Env:CHUNK_SIZE
python src\store.py   # rebuild the default 500/50 index afterwards
```

## Pipeline details

**Cleaning.** Raw PDF text had stray spaces and sentences broken by hard line wraps. The cleaner strips each line and rejoins wrapped lines so each numbered policy section becomes one paragraph (leave policy: 8 paragraphs, WFH: 8, travel: 7).

**Chunking.** Fixed-size character windows of 500 characters with 50 overlap, giving 9 chunks for these documents (15 with chunk size 300). Each chunk keeps its source file name.

**Retrieval.** The question is embedded with the same model and ChromaDB returns the k nearest chunks (default k = 3). Retrieval sanity check: for 5 direct questions the chunk containing the answer ranked **#1 in 5 of 5**.

**Generation.** The system prompt tells the model to use only the supplied context, reply exactly *"I don't know based on the provided documents."* if the answer is absent, avoid outside knowledge, and keep answers short with exact numbers. Temperature is 0 so test runs are repeatable.

## Testing

`tests/questions.csv` contains 25 questions:

| Category | Count | Description |
|---|---|---|
| direct | 15 | Answer is in one place in the documents |
| two_fact | 5 | Needs facts from different sections or files |
| unanswerable | 5 | Topic is not in the documents, so the correct output is a refusal |

Each question has an `expected_answer` and a `must_contain` keyword list used for automatic grading. A script runs every question and logs question, answer, retrieved chunks, pass/fail, notes, and the settings used (model, chunk size, overlap, top-k) to the SQLite table `results`.

Grading rules: unanswerable questions pass only if the model refuses; other questions fail if the model refuses or if a required keyword is missing (numbers are compared with commas removed).

### Results

Each experiment changed **one** thing from the baseline.

| Run | Change | Chunk / overlap | Top-k | Score | Accuracy |
|---|---|---|---|---|---|
| baseline | none | 500 / 50 | 3 | 24/25 | 96% |
| k5 | top-k 3 to 5 | 500 / 50 | 5 | 24/25 | 96% |
| chunk300 | smaller chunks | 300 / 50 | 3 | 24/25 | 96% |
| prompt_v2 | added multi-part instruction to prompt | 500 / 50 | 3 | 24/25 | 96% |

By category (baseline): 15/15 direct, 4/5 two-fact, 5/5 unanswerable. All five unanswerable questions were correctly refused in every run, including a tricky one (international travel allowance) where the topic is near, but not covered by, the documents.

### The failure I found

The same question failed in all four runs:

> "What happens after three late marks in a month and how many casual leave days do I get a year?"

It needs the late-marks rule (WFH policy, section 7) and the casual leave count (leave policy, section 1).

**Diagnosis.** I inspected the stored retrieved chunks for each failed run:

| Run | Late-marks chunk retrieved | Casual-leave count chunk retrieved |
|---|---|---|
| baseline | No | Yes |
| k5 | No | Yes |
| chunk300 | Yes | No |

This is a **retrieval failure, not a generation failure**. The question mixes two topics, so its single embedding sits closer to one topic and the other needed chunk is left out. The model behaved correctly each time: it answered the half it could see and said "I don't know" for the half it could not.

**What I tried.** Increasing top-k, reducing chunk size and changing the prompt wording (one at a time) did not fix it. With chunk size 300 the missing fact simply moved: late marks was found but the casual-leave count was lost. Prompt wording cannot fix a fact the model never received.

**Note on the grader.** Any answer containing "I don't know" is marked FAIL, so a half-correct answer scores as a failure. This is deliberate, because the answer was incomplete.

### Why distance cannot detect unanswerable questions

ChromaDB always returns the k nearest chunks. For the unanswerable question "What is the maternity leave policy?" the returned distances were 1.200 and 1.249, inside the range seen for correct answers (0.708 to 1.345). A distance cutoff would either block real answers or let this through, so the refusal is handled by the prompt.

## API

`POST /ask`

Request:
```json
{"question": "How much is the home internet allowance?"}
```

Response (HTTP 200, shortened):
```json
{
  "answer": "The home internet allowance is **Rs 1,000 per month**.",
  "sources": [
    {"file": "wfh_policy.pdf", "text": "... 5. Home Internet Allowance Employees on the hybrid schedule receive a home internet allowance of Rs 1,000 per month, paid along with the monthly salary. ..."},
    {"file": "wfh_policy.pdf", "text": "... 7. Attendance and Late Marks ..."},
    {"file": "travel_expense_policy.pdf", "text": "... 2. Hotel Limits ..."}
  ]
}
```

Example call:
```
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"How much is the home internet allowance?\"}"
```

Interactive docs are at `/docs` (screenshot: `docs/api_screenshot.png`).

The index must be built (`python src\store.py`) before starting the API. The server loads the embedding model at startup, so the first start takes a few seconds.

## Limitations

- **Tiny, clean corpus.** 3 documents, 9 chunks. One test question equals 4 percentage points, so small differences between runs are not meaningful. With k = 5 the system retrieves over half of all chunks, so k = 5 results say little about larger corpora.
- **Keyword grading** can mis-score correct answers (for example "eight" instead of "8") and short keys such as "8" could match by accident. Failures were read manually.
- **Compound questions** that span two topics can miss one needed chunk (the failure above).
- **PDF extraction** produced some glued words (for example "10days", "calendaryear"). Retrieval still worked, but I did not fix it.
- **Fixed-size chunking** can cut mid-word or mid-sentence.
- **Test questions were designed by me** for these documents, so the score is optimistic compared with real user questions.
- **No conversation memory, authentication, or rate limiting** in the API.
- Free-tier LLM rate limits apply; the test runner pauses 2 seconds between questions.

## What I would do next

1. **Query decomposition:** split compound questions into sub-questions, retrieve for each, then combine.
2. **Section-aware chunking:** split on policy section headings instead of fixed character counts.
3. **Test on larger, messier real-world PDFs** (headers, tables, scanned pages).
4. **Better evaluation:** an LLM-based or human-reviewed grader, and a larger test set including paraphrased and misspelled questions.
5. **Hybrid retrieval** (keyword plus embeddings) and a re-ranker.

## Security

The API key lives only in `.env`, which is excluded by `.gitignore`. Never commit it.