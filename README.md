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

- **PDF** is extracted with **PDFium** (`pypdfium2`). It replaced `pypdf`, which
  silently dropped whole headings from subsetted Arabic fonts; `pdfplumber` /
  `pdfminer` were also measured and return Arabic in *visual* (reversed) order,
  so they are not usable here.
- **PowerPoint / Word / plain text** are accepted and stored, but marked
  *Format not supported yet*; their extractors are stubs behind the same
  interface, each with a TODO describing what it must do.

### Arabic

PDFium reports a lam-alef ligature as its two letters in visual order — every
`الاصطناعي` would read `االصطناعي`. `repair_lam_alef()` puts them back:
the two letters of a ligature share one character box, whereas the definite
article `ال` is two glyphs with two boxes, so the repair is exact and never
touches ordinary text. Extracted text is stored in logical order and rendered
with `dir="auto"`.

### Honest reporting of what could not be read

Extraction never passes off a fragment as a full page:

- **Pages with no text layer** (a slide whose body is a screenshot, a scan, a
  diagram) are flagged `is_image_only`. The file's status becomes *Some pages
  have no text layer* with the count, and the reader says the page needs OCR.
  A file where **no** page has text is *No text layer*, as before. OCR is
  deliberately deferred — but a half-read file is never reported as complete.
- **Unmappable characters** are counted per file. Some PDFs embed subsetted
  fonts whose internal character tables are incomplete; the affected glyphs
  (usually decorative headings) cannot be recovered by any extractor without
  OCR, so the count is surfaced rather than hidden.

Uploads are written to `MEDIA_ROOT` (`media/` by default, git-ignored).

### Diagnosing an extraction

```bash
uv run python manage.py extraction_report                 # every uploaded file
uv run python manage.py extraction_report --file 1 --verbose-pages
uv run python manage.py extraction_report --reingest      # re-extract, then report
```

It lists, per page, how much text was stored and why a page is short — blank,
image-only, or suspiciously thin (which is what a real extraction bug looks
like).

## Tests

```bash
uv run python manage.py test          # everything (needs PostgreSQL)
uv run python manage.py test tests.test_provider   # no DB, no network
```
