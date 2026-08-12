# إحكام (Ihkam)

Intelligent agentic system for exam generation and review.
إحكام prepares the draft and surfaces imbalances. **The final decision always belongs to the instructor.**

Build status: **M1 complete** — M0 (skeleton, instructor auth, the `LLMProvider`
abstraction §3, the design system §5) plus courses, file upload and PDF text
extraction with page numbers preserved. Nothing from M2 onward exists yet.

## Requirements

- Python 3.12 (pinned in `.python-version`, installed by `uv`)
- [`uv`](https://docs.astral.sh/uv/) — the only package/environment manager used here
- PostgreSQL 14+ with the `pgvector` extension available

## Setup

```bash
uv sync                      # create the env and install locked dependencies
cp .env.example .env         # then fill in secrets — see below
uv run python manage.py migrate
uv run python manage.py runserver
```

## Environment

Everything secret lives in `.env` (git-ignored). See `.env.example` for the full
list: Django secret/debug/hosts, PostgreSQL connection, and the LLM block —
`LLM_PROVIDER` (`openai` | `gemini` | `airllm`), model names, API keys, and
`EMBEDDING_DIM`.

## The LLM seam (§3)

No module outside `agents/provider.py` may import an OpenAI / Gemini / AirLLM
SDK. Agents call `get_provider()` and use `complete()` / `embed()` only. That is
what makes the later switch to a local model a one-class change.

```bash
uv run python manage.py llm_ping                      # uses LLM_PROVIDER from .env
uv run python manage.py llm_ping --provider openai    # test one without editing .env
uv run python manage.py llm_ping --provider gemini
```

`AirLLMProvider` is a deliberate stub until Stage 5.

## Course material (M1)

An instructor creates a course, uploads a lecture file, and reads the extracted
text page by page. `courses/services/ingest.py` is the single entry point:
`extract_pages(fileobj, kind)` returns one `PageText` per page, in order, with
whitespace normalised and the **page number preserved** — every later citation
("source: page 14") is built from it.

- **PDF** is extracted (via `pypdf`).
- **PowerPoint / Word / plain text** are accepted and stored, but marked
  *Format not supported yet*; their extractors are stubs behind the same
  interface, each with a TODO describing what it must do.

**Known gap — scanned PDFs.** A photographed or scanned document has no text
layer. It does not crash: the file is marked *No text layer*, its page count is
still recorded, and the instructor is told to upload a text-based PDF. Text
recognition (OCR) is deliberately deferred.

Uploads are written to `MEDIA_ROOT` (`media/` by default, git-ignored).

## Tests

```bash
uv run python manage.py test          # everything (needs PostgreSQL)
uv run python manage.py test tests.test_provider   # no DB, no network
```
