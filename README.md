# RAG

A RAG (Retrieval-Augmented Generation) engineering project.

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # then fill in your API key(s)
```

## Run

```bash
python main.py
```

## Reindex

Rebuilds the local vector index from scratch (use after adding/removing PDFs
or changing chunking/embedding config):

```bash
python scripts/reindex.py
```

- Deletes and rebuilds `data/chroma/` and `data/parent_store/`.
- Source documents in `documents/` are **never** deleted.
- Runtime vector data is not committed to Git.
- Use `--yes` to skip the confirmation prompt (e.g. in CI).

## Environment variables

See `.env.example` for required keys:

| Variable | Provider |
|---|---|
| `GEMINI_API_KEY` | Google Gemini |
| `GROQ_API_KEY` | Groq |
| `MISTRAL_API_KEY` | Mistral AI |