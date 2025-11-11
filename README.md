# Text-to-SQL Chatbot (Simplified Single-Path Workflow)

A clean, focused Text-to-SQL app: upload Excel/CSV files, we build a SQLite DB, and answer natural language questions via a single LangGraph workflow with validation and self-correction.

## Workflow

```
User Question
     ↓
┌─────────────────────────────────────┐
# Text-to-SQL Chatbot (Simplified Single-Path LangGraph Workflow)

This repository contains a focused Text-to-SQL application. Upload Excel/CSV files, the app builds a SQLite database, and answers natural language questions using a single LangGraph workflow with validation and self-correction.

## Our approach

When a user asks a question, the system routes the request through a single LangGraph workflow that handles classification, generation, validation, self-correction, and execution.

```
User Question
     ↓
┌─────────────────────────────────────┐
│  LangGraph Workflow (Single Path)   │
├─────────────────────────────────────┤
│  1. Classifier Agent                │
│     ↓                               │
│  2. SQL Generator Agent             │
│     • Fine-tuned model (SIMPLE)     │
│     • Groq LLM (COMPLEX)            │
│     ↓                               │
│  3. Validator Agent                 │
│     ↓                               │
│  4. Self-Corrector Agent (Groq)     │
│     ↓                               │
│  5. Execute                         │
└─────────────────────────────────────┘
```

Key ideas:
- A single, deterministic path simplifies maintenance and observability.
- Simple queries are served by a local fine-tuned model (free/unlimited).
- Complex queries, or any fallback/correction steps, use the Groq LLM for higher accuracy.

## Quickstart

1) **Clone the repository**
```bash
git clone https://github.com/rahul-1809/Ask2SQL.git
cd Ask2SQL
```

2) **Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) **Configure your API key**
```bash
# Set your Groq API key (required for complex queries and corrections)
export GROQ_API_KEY="your-groq-api-key-here"

# Get your free API key from: https://console.groq.com/keys
```

4) **Run the app**
```bash
chmod +x run.sh
./run.sh
```
Visit http://127.0.0.1:5000

> **⚠️ Security Note:** Never commit your API keys to Git. Always use environment variables or `.env` files (which are gitignored).

## Important files

- `app/app.py` – Flask app (uses the simplified single-path workflow)
- `app/sql_workflow_simplified.py` – LangGraph workflow: Classifier → Generator → Validator → Corrector
- `app/query_classifier.py` – Rule-based SIMPLE/COMPLEX classifier
- `app/sql_validator.py` – Schema checks and aggregation/GROUP BY rules
- `app/database_utils.py` – CSV/Excel → SQLite, schema extraction, safe execution, table summaries
- `app/templates/` – HTML templates (UI)
- `models/` – Optional: place fine-tuned model in `models/fine_tuned_text2sql_codet5/`

Deprecated modules (kept as stubs): `model_utils.py`, `groq_text2sql.py`, `sql_workflow.py`, `config.py`.

## Notes & behavior

- The generator prefers a local fine-tuned CodeT5 model for SIMPLE queries. If a LoRA adapter is present, it will be applied to a base model using PEFT. If no fine-tuned model is available, the system falls back to a base model.
- COMPLEX queries and all correction attempts use the Groq API when `GROQ_API_KEY` is configured.
- SQL validation ensures table/column correctness and enforces GROUP BY when aggregations are present. Only safe SELECT statements are executed.

## Interactive features

- Table summaries: after uploading files, the UI shows table-level summaries (columns, sample rows, numeric stats).
- Query explanations: each generated SQL includes a concise, human-readable explanation (Groq) to help users learn and verify results.

## Minimal test

```bash
python test_workflow_min.py
```

## Troubleshooting

- If Flask does not start, ensure no other process is using port 5000 and run `./run.sh`.
- If the fine-tuned model is missing, the system falls back to the base model (`Salesforce/codet5-small`).
- Loading LoRA adapters requires `peft` installed in your environment.

## License
MIT
