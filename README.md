# إحكام (Ihkam)

Intelligent agentic system for exam generation and review.
إحكام prepares the draft and surfaces imbalances. **The final decision always belongs to the instructor.**

Build status: **M2 complete** — M0 (skeleton, instructor auth, the `LLMProvider`
abstraction §3, the design system §5), M1 (courses, file upload and PDF text
extraction with page numbers preserved, plus OCR), and M2 (topic extraction, the
instructor topic-review screen, and chunks + embeddings in `pgvector`). Nothing
from M3 onward exists yet — there is no retrieval, blueprint or agent.

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

### OCR (pages with no text layer)

Pages flagged `is_image_only` are transcribed by a **vision-language model**,
not a classic OCR engine: the material is Arabic + English, often on the same
page, and VLMs read Arabic correctly where traditional engines return reversed
or disconnected glyphs. This runs **synchronously during upload** — no queue,
no background worker.

`agents/ocr.py` is the seam, shaped exactly like `agents/provider.py`:
`OCRProvider.ocr_page(image) -> OCRResult`, selected by `OCR_PROVIDER`.
`GeminiOCRProvider` reuses the existing Gemini key; `PaddleVLProvider` is the
local phase-2 stub (**full server model**, Apache-2.0 — not the mobile
variant). Nothing outside that module imports a vision SDK.

- **One image per request.** Batching pages into one call measurably degrades
  transcription quality. Pages are sent concurrently (`OCR_CONCURRENCY`) only
  to shorten the wait.
- **OCR'd pages are marked** `source="ocr"` and labelled in the reader as a
  model transcription, so later milestones can weigh that evidence differently
  from a real text layer.
- **Empty or failed transcriptions are never stored.** The page keeps its
  "no readable text" flag — an unread page is not a blank page.
- **`OCR_MAX_PAGES_PER_FILE`** caps the work per upload. A file over the cap is
  not rejected: the first pages are read, the rest stay flagged, and the file
  says so.

```bash
uv run python manage.py ocr_probe --file 1 --pages 8,13   # see a transcription
uv run python manage.py ocr_probe --image-only --limit 4
```

> **API quota is the real constraint.** A free Gemini key is limited to a few
> requests per *minute* and only ~20 per *day*. A per-minute limit is waited
> out automatically (the server's own `retryDelay` is respected); a per-day
> quota is not — it stops the pass immediately and records why, rather than
> hanging the upload for a quota that resets tomorrow. A 21-page deck needs a
> billed key to finish in one upload.

### Honest reporting of what could not be read

Extraction never passes off a fragment as a full page:

- **Pages with no readable text** — not read by extraction *or* OCR — stay
  flagged `is_image_only`. The file's status becomes *Some pages have no text
  layer* with the count, and the reader says so on the page itself. A file
  where **no** page has text is *No text layer*. A half-read file is never
  reported as complete.
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

## Topics and chunks (M2)

### The topic list is a draft until the instructor confirms it

`courses/services/topics.py` sends the course's **readable** pages through
`get_provider()` (never an SDK directly), asks for JSON, and validates it twice
before a row is written:

1. **Shape**, with Pydantic. A malformed answer is retried **once**, then
   surfaced as an error — unvalidated output never reaches the database. A call
   that never got to the model (no key, no credit, rate limit) is reported as
   *that*, not as a bad answer, and is not retried.
2. **Truth about the material**, in `_resolve_file` / `_resolve_span`. A file
   name must be a file this course really has, and a page span must overlap the
   pages actually sent. A span reaching past them is narrowed to the real part;
   one that is entirely invented is dropped rather than stored. A hallucinated
   citation is the failure this system exists to prevent.

Then the instructor reviews it at `/courses/<id>/topics/` — rename, merge two,
delete, add by hand, or mark **"not taught in lectures"**. This is a required
product step, not a convenience screen.

- **Excluded topics never flow downstream.** `Topic.objects.included()` and
  `Chunk.objects.usable()` are the only ways anything later reads them.
- **Deleting a chapter promotes its sub-topics** rather than deleting them —
  removing a heading is not a request to lose what is filed under it.
- **Merging keeps everything both topics knew**: the combined page span (when
  both cite the same file), de-duplicated terms/formulas/examples, re-parented
  sub-topics, re-pointed chunks. The survivor keeps its own name.
- **Re-extraction replaces the whole list** and says so before it runs, behind
  a separate "Replace all topics" button.

### Content source discipline

Only readable pages contribute — `ExtractedPage.is_readable` is the single
definition, shared by extraction and chunking. A page with no text layer that
OCR could not recover contributes nothing to either; letting its leftover slide
number into the prompt would invite a topic invented out of a page number.
OCR'd pages **are** used, labelled `(OCR transcription)` in the prompt, and the
prompt tells the model that text is a reading of a picture rather than a text
layer — not more authoritative than one.

### Chunks + embeddings

`courses/services/chunking.py` splits each readable page into passages and
stores one embedding per passage in `chunks` (`pgvector`), through the same
`embed()` the `llm_ping` command uses.

- **A passage never crosses a page boundary.** Every citation is only as good
  as the page number on the passage it quotes.
- **Provenance is recorded** per chunk (`text_layer` vs `ocr`), so later
  milestones know what they are quoting.
- **The width is checked against `EMBEDDING_DIM` before anything is stored.** A
  provider quietly returning its native size would make every stored vector
  incomparable, and the damage would not appear until retrieval in M3.
- Chunks are attached to topics **deterministically, by page span** — the
  narrowest claim wins, and a passage no topic claims keeps `topic = None`.

Chunking runs at the end of upload. `EMBEDDINGS_ENABLED=false` skips it (the
test suite does this, so the suite spends no API calls):

```bash
uv run python manage.py build_chunks --course CS310   # or --all
```
