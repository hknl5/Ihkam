# إحكام (Ihkam) — Build Execution Plan

> Intelligent agentic system for exam generation and review.
> The system prepares the draft and surfaces imbalances. The final decision always belongs to the instructor.

This plan is written to be handed **directly to a coding agent**. Each milestone is self-contained, testable, and measurable. Build order follows the three agents (1A → 2A → 3A), because each agent is the natural seam where you can stop, run something real, and verify before moving on.

---

## 0. How to read this plan

- **Stack is fixed:** Django + Django REST Framework backend, PostgreSQL + `pgvector`, server-rendered Django templates for the UI (no separate React app — the original brief mentioned React, but for a vibe-coded MVP that one person can build and test, Django templates + a light sprinkle of HTMX/Alpine is far faster to iterate on). If you later want React, the API boundary defined here keeps that door open.
- **LLM is swappable from day one.** Every call to a model goes through a single `LLMProvider` interface. Start with OpenAI or Gemini API keys to confirm the system works end-to-end, then swap to a local model via **AirLLM** by writing one new provider class. No business logic changes.
- **Each milestone has:** a goal, what you build, a **manual test you can run yourself**, and a **measurable success check**. Do not move to the next milestone until the current one passes.
- **Every phase in the original brief maps to a milestone here.** Nothing from the brief is dropped; some later phases (LMS export, question bank polish) land in the final milestones.

---

## 1. Architecture at a glance

```
┌─────────────────────────────────────────────────────────────┐
│  Django Templates (server-rendered) + HTMX + Alpine.js       │
│  Courses · Upload · Blueprint · Review · Compare · Bank       │
└───────────────────────────┬─────────────────────────────────┘
                            │  (Django views / DRF endpoints)
┌───────────────────────────▼─────────────────────────────────┐
│  Django app: ihkam                                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ Agent 1A   │→ │ Agent 2A   │→ │ Agent 3A   │             │
│  │ Analyze &  │  │ Generate   │  │ Review &   │             │
│  │ Plan       │  │ Questions  │  │ Balance    │             │
│  └────────────┘  └─────▲──────┘  └─────┬──────┘             │
│                        └── reject ─────┘                    │
│  LLMProvider (OpenAI / Gemini / AirLLM-local)               │
│  Deterministic engine (scoring, weights, timing, exports)   │
│  Retrieval (pgvector chunks)                                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  PostgreSQL + pgvector                                       │
│  courses · files · chunks · topics · exams · blueprints ·    │
│  questions · reviews · forms · question_bank                 │
└─────────────────────────────────────────────────────────────┘
```

**Core principle carried into code:** what can be computed exactly (score sums, topic weights, timing, workflow control, export files) is done in **plain Python**. Only what needs language understanding (extraction, generation, clarity review, semantic similarity) is given to the **LLM**. This is a hard rule — see the responsibility split in Milestone tables.

---

## 2. Repository layout

```
ihkam/
├── manage.py
├── pyproject.toml               # replaces requirements.txt (uv-managed)
├── uv.lock
├── .env.example
├── config/                      # Django project (settings, urls, wsgi)
│   ├── settings.py
│   └── urls.py
├── accounts/                    # instructor auth
├── courses/                     # courses, files, chunks, topics
│   ├── models.py
│   ├── views.py
│   ├── services/
│   │   ├── ingest.py            # file → text → chunks
│   │   └── retrieval.py         # pgvector query
├── exams/                       # exams, blueprints, questions, forms, reviews
│   ├── models.py
│   ├── views.py
│   ├── services/
│   │   ├── blueprint.py         # deterministic scoring/weight checks
│   │   ├── forms.py             # form assembly + convergence checks
│   │   └── export.py            # PDF now, LMS formats later
├── agents/                      # the three agents + provider
│   ├── provider.py             # LLMProvider interface + implementations
│   ├── ocr.py                  # OCRProvider seam (Gemini vision / PaddleOCR-VL)
│   ├── prompts/                # versioned prompt templates
│   ├── analyze.py              # Agent 1A
│   ├── generate.py             # Agent 2A
│   ├── review.py               # Agent 3A
│   └── orchestrator.py         # closed correction loop 2A ⇄ 3A
├── bank/                        # approved question bank
├── templates/                   # Django templates (UI)
├── static/                      # CSS design system, JS
└── tests/
```

---

## 3. The LLM provider abstraction (build this FIRST, before any agent)

Everything depends on this. It is what lets you start on API keys and later move to a local model with AirLLM.

```python
# agents/provider.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    text: str
    raw: dict

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, *, json_mode: bool = False,
                 temperature: float = 0.2) -> LLMResponse: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class OpenAIProvider(LLMProvider):   # phase 1 default
    ...
class GeminiProvider(LLMProvider):   # alternate phase 1
    ...
class AirLLMProvider(LLMProvider):   # phase 2 local, drop-in
    ...

def get_provider() -> LLMProvider:
    # reads settings.LLM_PROVIDER, returns the right instance
    ...
```

**Rules:**
- Agents **never** import OpenAI/Gemini/AirLLM directly. They call `get_provider()`.
- All prompts return **structured JSON** (`json_mode=True`), validated with a Pydantic schema before use. If validation fails, retry once, then surface an error — never pass unvalidated model output downstream.
- Embeddings go through `embed()` too, so the local switch covers retrieval and similarity as well.
- Keep `LLM_PROVIDER`, model names, and keys in `.env`.

**OCR follows the same pattern.** `agents/ocr.py` defines an `OCRProvider` seam shaped exactly like `LLMProvider`, selected by `OCR_PROVIDER`: Gemini vision now, with a local **PaddleOCR-VL** (full server model) stub in place for later.

**Test for this milestone:** a tiny management command `python manage.py llm_ping` that runs one completion and one embedding and prints the result. Switch the env var between `openai` and `gemini` and confirm both work. **Success = both providers return valid JSON and a vector of the expected dimension.**

---

## 4. Milestones

The build is grouped into **five stages** mapping to your "test-as-you-go, agent-by-agent" requirement.

| Stage | Milestones | What becomes testable |
|-------|-----------|-----------------------|
| **S0 Foundation** | M0–M2 | Project runs, files ingest, topics extracted & editable |
| **S1 Agent 1A** | M3–M4 | Retrieval works; blueprint builds and validates |
| **S2 Agent 2A** | M5–M6 | Questions + answer keys generate within scope |
| **S3 Agent 3A** | M7–M8 | Review loop rejects/regenerates; quality gate holds |
| **S4 Forms & Ship** | M9–M12 | Two forms, convergence checks, review UI, export, bank |

---

### STAGE 0 — Foundation

#### M0 · Project skeleton + auth + design system
**Goal:** a running Django app an instructor can log into, with the visual language in place.

Build:
- Django project, PostgreSQL with `pgvector` extension enabled (`CREATE EXTENSION vector;`).
- `accounts` app: instructor sign-up/login (Django's built-in auth is fine).
- The `LLMProvider` abstraction from §3 and the `llm_ping` command.
- Base template + the CSS design system from §5 (tokens, type, components). Every later screen inherits this.

**Manual test:** run server, sign up, log in, land on an empty "My courses" dashboard. Run `llm_ping`.
**Success check:** auth works; `llm_ping` returns valid output on both API providers; base layout renders correctly on mobile and desktop with visible keyboard focus.

---

#### M1 · Course + content upload + text extraction
**Goal:** instructor creates a course and uploads material; the system extracts clean text. (Brief Phase 2, steps 1–2.)

Build:
- `Course` model: name, code, level, content language.
- `SourceFile` model + upload. Support **PDF** first (MVP scope); stub PowerPoint/Word/plain-text behind the same interface for later.
- **Scope:** PDF text extraction via PDFium; automatic OCR for pages that are image-only, mixed (text + image body), or have defective-font junk; OCR runs synchronously on upload with parallel per-page calls, behind a swappable `OCRProvider`.
- `courses/services/ingest.py`: extract text from PDF, normalize whitespace, keep page numbers (needed later for "source: Slide 3, Chapter 14"–style citations).

**Manual test:** create a course, upload a real lecture PDF, view extracted text page by page.
**Success check:** text extraction is legible and complete for a text-based PDF; page references are preserved. Scanned, mixed, and defective-font pages come back readable through OCR.

> **M1 technical decisions:**
> - **PDFium** chosen over pypdf/PyMuPDF — Apache-2.0 licence and noticeably better extraction on real lecture files.
> - **Lam-alef ligature repair** on extracted Arabic text.
> - **Gemini OCR** for image-only, mixed, and defective-font pages.
> - **Defective-font pages keep their existing text layer** if OCR returns <60% of its length — a guard against losing body text to a partial transcription.
> - **Greek letters excluded** from junk detection (they are legitimate content, not font damage).

---

#### M2 · Topic extraction + instructor review of topics
**Goal:** turn uploaded content into an editable topic list. (Brief Phase 2, step 3 + the mandatory human-review step.)

Build:
- `Topic` model: name, parent chapter, source file, page span, `excluded` flag.
- LLM call (via provider) that extracts chapters, sub-topics, terms, definitions, formulas, examples → validated JSON → `Topic` rows.
- **Topic review screen** — this is a required product step, not optional: instructor can rename, merge two topics, delete, add, or mark "not taught in lectures." Nothing the model extracted is trusted until the instructor confirms it.

**Manual test:** upload a chapter, see extracted topics, rename one, merge two, exclude one, add one manually, save.
**Success check:** edits persist; excluded topics never appear downstream; extracted structure matches the document's actual sections on 3 sample courses (theoretical, math-heavy, programming) — this is also your first entry in the validation plan (§6).

> **Chunking note:** during ingest, split text into passages and store an embedding per passage in a `chunks` table (`pgvector`). This is done now even though retrieval is tested in M3, because chunks must exist before Agent 1A can retrieve. Store `chunk(text, embedding, source_file, page, topic_fk_nullable)`.

> **M2 technical decisions:**
> - **`Topic` carries the detail it was extracted with** — `key_terms`, `definitions`, `formulas`, `examples` as JSON alongside name / parent / source file / page span / `excluded`. The prompt asks for all six things the brief names, and discarding four of them would only mean re-extracting them in M5.
> - **Parent chapter is a self-FK**, so a chapter is a topic with no parent. Deleting a chapter **promotes** its sub-topics rather than taking them with it.
> - **Page spans are verified against the pages actually sent**, not just parsed. A span reaching past them is narrowed to the real part; an entirely invented one is dropped. A file name the course has not got yields no citation at all.
> - **Retry once, but only for a malformed answer.** A call that never reached the model (no key, no credit, rate limit) is surfaced as itself — reporting it as "bad JSON" sends the instructor to the wrong place, and a second call would fail identically.
> - **`ExtractedPage.is_readable` is the one definition** of what may contribute, shared by extraction and chunking. OCR pages are used and labelled `(OCR transcription)`; unread pages contribute nothing.
> - **A chunk never crosses a page boundary**, and records `source` (`text_layer` / `ocr`). Embedding width is checked against `EMBEDDING_DIM` before anything is stored.
> - **Chunks are linked to topics deterministically by page span** (narrowest claim wins), so `Chunk.objects.usable()` can honour an exclusion without a model in the loop.
> - **Re-extraction replaces the whole list**, behind a separate button that says so — an instructor's edits are never silently overwritten.

---

### STAGE 1 — Agent 1A: Analyze & Plan

Agent 1A **does not write questions.** It understands content, builds the blueprint, and retrieves the reference passages each future question will be grounded in.

#### M3 · Retrieval (RAG) works
**Goal:** given a topic, return only the relevant passages — not the whole course.

Build:
- `courses/services/retrieval.py`: embed the topic/query, search `chunks` by vector similarity in `pgvector`, return top-k passages with page refs.
- A debug view: type a topic, see the passages the system would hand to the generator.

**Manual test:** query a specific topic, inspect returned passages.
**Success check:** returned passages are actually about the queried topic on your 3 sample courses; unrelated chapters do **not** leak in. Eyeball precision on ~10 queries.

---

#### M4 · Blueprint builder + deterministic validation
**Goal:** produce and validate the exam blueprint before any question exists. (Brief Phase 3.)

Build:
- `Exam` model: type (quiz/mid/final), total score, question count, duration, language, number of forms.
- `Blueprint` + `BlueprintRow`: topic, score %, question type, count, level.
- `exams/services/blueprint.py` — **pure Python, no LLM.** Validates:
  - question counts and scores sum to the required totals,
  - topic weights add up (no over/under),
  - no selected topic left with zero questions,
  - score distribution matches topic weights (e.g. a 30%-weighted topic in a 40-mark exam = 12 marks).
- Blueprint editing UI: instructor adjusts rows by hand; validation re-runs live.
- Agent 1A wiring: for each blueprint row, call retrieval (M3) and attach reference passages. Output = **blueprint + reference passages per planned question**, ready to hand to Agent 2A.

**Manual test:** define an exam, auto-build a blueprint, deliberately break a weight (make it sum to 110%), confirm the validator flags it; fix it; confirm it passes.
**Success check:** every arithmetic check catches its error class; a valid blueprint emits one reference-passage bundle per planned question.

> ✅ **End of Stage 1: Agent 1A is complete and independently testable.** You can now analyze content and produce a validated, grounded plan without generating a single question.

---

### STAGE 2 — Agent 2A: Generate Questions

Agent 2A receives **one blueprint item at a time** (topic, type, level, score, expected time, reference passages) and returns the question, its options, answer, explanation, and source. It **over-generates** to provide alternatives.

#### M5 · Question generation within scope
**Goal:** generate questions strictly from the retrieved passages. (Brief Phase 4, agent 2A.)

Build:
- MVP question types only: **MCQ, True/False, Short Answer, simple text-based numeric problems.** (Diagrams, symbolic math, code analysis, matching, image-based → deferred, per brief.)
- `agents/generate.py`: input one blueprint item + passages → validated JSON with `stem, type, options[], correct, explanation, source_ref`. Generate **N+extra** candidates per item for alternatives.
- `Question` model with status field: `candidate → approved/rejected`.

**Manual test:** run generation for one blueprint row, read the candidates, check each cites a real passage/page.
**Success check:** ≥ target share of candidates are on-topic and derived from the uploaded material (not general knowledge). Any question citing content outside the passages is a failure — track the rate.

> **M5 technical decisions:**
> - **Over-generation is `ceil(N × 1.5)`**, behind the named `over_generated_count()` helper so M7 can tune it once the review loop shows how many candidates survive. A row of 1 yields 2 — there is always an alternative to reject the first in favour of.
> - **Passages are labelled `P1…Pn` in the prompt** and every candidate must return one as `source_ref`. The label is resolved back to the supplied `Passage` object; a citation that resolves to nothing is **dropped and counted**, never stored. That count is the M5 success metric.
> - **The grounding instruction is one named constant** (`GROUNDING_RULE`), stated in both halves of the call and asserted by a test — if it ever stops reaching the model, generation silently becomes general knowledge with citations attached.
> - **Per-type validation lives in the Pydantic model**: an MCQ whose `correct` is not one of its options is not a flawed candidate but an unusable one, so it is a validation failure that gets the single retry. Retry once for a malformed answer; a call that never reached the model is reported as itself (the M2 rule).
> - **A candidate of the wrong type is dropped**, not kept — the blueprint decided the type, and a short answer in an MCQ row spends marks the plan did not allocate.
> - **`from_ocr` propagates from the cited passage** onto the `Question`, so M6 and review know the question is quoting a transcription of a picture.
> - **Deferred types are refused by name before any call** (`UnsupportedQuestionType`), never silently downgraded to MCQ.
> - **`Question` lives in `exams/`, not `agents/`** — it is exam content, not agent machinery — and its `blueprint_row` link is `SET_NULL` so re-planning a paper does not delete questions already approved.

---

#### M6 · Answer key generated *with* the question
**Goal:** the answer model is produced in the same step as the question, never after. (Brief Phase 5.)

Build:
- Extend generation so each candidate carries its answer key by type:
  - objective → direct key,
  - short answer → model answer + required elements,
  - numeric → solution steps + final answer + suggested per-step marks,
  - (essay key format defined now, essay questions themselves are post-MVP.)
- Store answer key on the `Question` row so review (Stage 3) can inspect question + answer together in one pass.

**Manual test:** generate a numeric question, confirm it comes back with steps and a mark split already attached.
**Success check:** 100% of generated candidates arrive with a type-appropriate answer key; none require a second call to produce the key.

> **M6 technical decisions:**
> - **The key is validated in the same object as the stem** (`CandidateOut.answer_key` → `agents/answer_key.py`). A short answer with no required elements, or a numeric with no steps, fails validation and is re-asked by the existing single retry — never stored keyless for someone to complete later.
> - **Objective keys are formalised, not demanded.** `correct` and the options already are the key, so an `ObjectiveKey` is built from them when the model does not repeat itself. The open types must supply theirs.
> - **The model splits the marks; Python only adds them up.** `check_mark_sum` compares the per-step marks with the question's marks in `Decimal`, and on a mismatch **flags** (`mark_sum_ok=False` plus a note) rather than rescaling — the first hook toward M7's deterministic maths checker. A flagged candidate is kept and reported, not dropped: the stem is not the arithmetic's fault.
> - **Stored as JSON plus one queryable column.** `Question.answer_key` holds the typed key (marks as strings, so half marks survive the round trip); `Question.mark_sum_ok` is a nullable boolean so review can *query* for the questions whose marks do not add up, and "not checked" stays distinct from "checked and wrong".
> - **The essay key format exists before essay questions do.** `EssayKey` (rubric criteria + weights) validates and round-trips now; `MVP_TYPES` still excludes essay, so none is generated.

> ✅ **End of Stage 2: Agent 2A is complete and independently testable.** Feed it a blueprint item, get grounded questions + answer keys back.

---

### STAGE 3 — Agent 3A: Review & Balance

Agent 3A checks each question but **does not rewrite it.** A rejected question goes back to 2A with a specific note for a fresh alternative. This closed loop is the heart of the quality story.

#### M7 · Quality review checks
**Goal:** automatically catch bad questions before the instructor ever sees them. (Brief Phase 6.)

Build:
- `agents/review.py` checking, per question:
  - **content link** — in scope, based on uploaded material, no outside info,
  - **clarity** — unambiguous, single interpretation, appropriate question verb,
  - **answer correctness** — one answer consistent with source (leave a hook for a later deterministic math checker),
  - **option quality** — exactly one correct, no silly distractors, no meaning duplicates, correct option not conspicuously longest,
  - **level match** — produced level vs requested level (the brief's worked example: a "compute Precision from a matrix" request that came back as "define Precision" must be rejected as too easy).
- Each rejection carries a **reason note**.

**Manual test:** hand-craft or generate a too-easy question, run review, confirm rejection + reason.
**Success check:** the level-mismatch example and an out-of-scope example are both caught with correct reasons.

> **M7 technical decisions:**
> - **The split is by what can be computed, not by what is convenient.** Python decides: exactly one correct option, no repeated option, the correct option is not conspicuously the longest, the answer key is present and usable for its type, and (via M6) the mark split adds up. The model decides only what needs language understanding: content link, clarity, answer consistency, level match, distractor plausibility. Sending a countable thing to a model would make a fixed answer probabilistic.
> - **Every rejection carries a `requirement`, not just a `reason`.** "Bad question" is not something a generator can act on. When the model fails a check without saying what to do, a per-check default instruction is filled in (`DEFAULT_REQUIREMENTS`) so M8 always has something to hand back.
> - **All checks run, even after the first failure.** One review, one complete set of notes, one regeneration — otherwise 2A fixes the level, regenerates, and only then discovers the scope problem.
> - **The maths hook is `MATH_CHECKERS`.** M6's `mark_sum_ok` is surfaced here as its first member, never recomputed, so one place in the system decides whether a mark split adds up. A later symbolic evaluator registers into the same list without touching `review_question`.
> - **Option normalisation strips edge punctuation only.** Stripping punctuation everywhere collapsed `(x y)′` and `x + y` into the same string and reported two different Boolean expressions as one option repeated — caught on the real discrete maths course. Interior symbols are content, not presentation.
> - **Nothing is stored and nothing is rewritten.** `ReviewOut` has no field a corrected question could arrive in, and review writes no rows; M8 decides what to do with a verdict.

---

#### M8 · Closed correction loop (orchestrator)
**Goal:** wire 1A → 2A → 3A into one automated pass with regeneration. (Brief agent workflow.)

Build:
- `agents/orchestrator.py`: for each blueprint item, generate → review → on reject, return to 2A **with the note** for an alternative → repeat until pass or a max-retries cap. Passing questions form the approved pool.
- Persist every attempt + note (traceability; also feeds success metric "problems caught before approval").
- Guardrail: cap retries so a stubborn item surfaces to the instructor as "needs manual attention" rather than looping forever.

**Manual test:** run a full blueprint end-to-end; watch the log show rejects returning to 2A and eventually passing.
**Success check:** every rejected question re-enters 2A with its note and nothing advances to form-building until it passes review or hits the manual-attention cap.

> **M8 technical decisions:**
> - **The loop fills a gap; it does not chase a candidate.** Regenerating each rejected question individually is the obvious design and the wrong one: a row of 5 that got 6 candidates and passed 5 is *finished*, and the sixth rejection is not a problem. The condition is `passing < N`, counted per row, and a gap-fill round asks only for the shortfall — over-generated the same way a first batch is, so a row short by one still gets two tries. Surplus passes are kept as the alternatives an instructor rejects the first choice in favour of.
> - **The notes reach the model, and all of them do.** `format_rejections` puts Agent 3A's own sentences — reason *and* requirement — plus the rejected stems into the next 2A prompt, immediately before the grounding rule. Every round's notes are carried, not just the last one's: a one-round memory produces a replacement that fixes the level and reintroduces the scope fault.
> - **Three gap-fill rounds, then a human.** A row the model cannot satisfy is usually a row whose material does not support the question the blueprint asks for; no number of retries fixes that. The item stops, is marked **needs manual attention**, and keeps whatever partial pool it earned.
> - **An outage is not a rejection.** `GenerationCallFailed` / `ReviewCallFailed` were split out of their parent errors so the loop can tell "the call never reached the model" from "the answer was unreadable" without matching on a message. A round only counts once it finished, so an outage costs the item no retry, and a whole-plan run stops rather than reporting the same outage against every remaining row.
> - **Every attempt is written down, including the ones that never became questions.** `ItemRun` + `QuestionAttempt` store passed, rejected and dropped-before-review alike, with the notes verbatim. M5 drops an ungrounded candidate silently by design; here that drop is what explains a short row, so `attempts == approved + rejected + dropped` is a property the suite holds.
> - **Passing review is not approval.** Stored questions keep status `candidate`, exactly as M5 stores them — review passing is إحكام's opinion and approval is the instructor's act. What M8 guarantees is only that nothing unreviewed can reach M9: the pool is read through the attempts that passed, not through `Question`.
> - **Re-running an item updates its log rather than adding a second one**, and a passing candidate whose stem is already stored reuses that row. Re-running costs model calls, not a duplicated pool.

> ✅ **End of Stage 3: all three agents work together in a closed loop.** This is the functional core of إحكام.

---

### STAGE 4 — Forms, review UI, export, bank

#### M9 · Form assembly (A & B) from one blueprint
**Goal:** build multiple forms from a **single blueprint** so they're similar by construction, not by luck. (Brief Phase 7.)

Build:
- `Form` model; `exams/services/forms.py` distributes approved questions across forms so each matches on: question count, score distribution, topic + type distribution, cognitive-level spread, count of multi-step and numeric items, expected time. Start with **rules + simple scoring** (LLM not needed here); leave a seam for an optimizer later.
- MVP cap: **two forms.**

**Manual test:** build Form A and Form B, eyeball that both cover the same topics with the same weights.
**Success check:** the deterministic distribution matches on all listed dimensions within tolerance.

> **M9 technical decisions:**
> - **The blueprint row is the unit of assembly, so seven of the eight dimensions cannot drift.** A row is one topic, one type, one level, one count, one price; every form takes `row.count` questions out of that row's pool. Count, marks, topic, type, level, multi-step and numeric therefore match *exactly* — their tolerance is zero and it is earned, not chosen to make the check pass. "Similar by construction" is this, stated concretely.
> - **Only expected time is scored, because only it is free.** Two MCQs on the same topic at the same level cost the same marks and take different minutes, so which surplus question goes to A is a real choice. `estimate_minutes` counts what can be counted — type, level, words to read, steps in the key — and never asks a model how hard a question is; that would be an opinion dressed as a number, which M10's honesty rule forbids.
> - **The allocator is a seam with a documented contract**, not a strategy baked into the caller. `greedy_balanced` (longest-processing-time first, deterministic ties) is passed to `distribute` as `allocate=`; an optimizer that balances *across* rows replaces it without touching the models, the reporting, or the screens.
> - **A shortfall is concentrated on the later form, and reported per row.** Two papers each missing one question are two papers that cannot be sat; a complete Form A and a Form B short by two is a finished paper plus a countable thing to generate. Each shortfall names the form, the topic, the count and both ways out — and `save_assembly` refuses to write an incomplete assembly at all, so a paper with a hole in it cannot reach a screen or an export.
> - **Sharing is a permission, not a fallback.** With sharing on, the allocator still fills both forms from distinct questions first and reuses one only once the pool runs out. A row too thin for even one paper is still short, and says so instead of telling the instructor to enable a setting that is already on.
> - **The sharing option is revealed by the same code that decides it applies.** A one-form exam is never asked how its forms relate; `_wants_multiple_forms` answers that for the first paint and for the HTMX endpoint alike, so the screen cannot offer an option the save path would ignore.
> - **Marks are priced per slot, not per question.** A 14-mark row over 3 questions is 4.67 / 4.67 / 4.66 by the same largest-remainder rule `auto_build` uses; three rounded shares would put the form on 14.01 and make the paper disagree with the blueprint it was built from.

---

#### M10 · Convergence checks + comparison screen
**Goal:** measure and display how close the forms are, honestly. (Brief Phase 8.)

Build:
- Checks: topic **coverage**, expected **difficulty** (proxy indicators: solution steps, length, formula presence, option complexity), expected **time** (per-question estimate summed vs limit), answer **leakage** (one question revealing another's answer), question **similarity** (semantic, via embeddings).
- Comparison screen: side-by-side table (counts, scores, per-chapter %, direct/applied/analytical counts, expected time) + إحكام notes ("Form B has one extra multi-step question"; "Q12B needs a clarity check"; "Q8A may help answer Q17A").
- **Naming discipline (hard rule):** label results "expected difficulty," never "actual difficulty." Never claim two forms are "X% equivalent" — there's no measurement basis until students sit the exam. Show real indicators as-is.

**Manual test:** compare two forms, verify the notes point to genuine issues you can confirm by eye.
**Success check:** leakage and high-similarity pairs are detected on a seeded test; no percentage-equivalence claim appears anywhere in the UI.

> **M10 technical decisions:**
> - **The naming rule is code, not a habit.** `dishonest_claims()` lists the claims this milestone may not make ("actual difficulty", any percentage attached to equivalence or similarity, any equivalence *score*), and the suite runs it over the serialised report **and** over the rendered HTML. A hard rule that lives only in a docstring is a hope.
> - **Expected difficulty is four proxies, never one index** — worked steps, words to read, formula presence, options beyond four. Combining them needs weights nobody can justify and hides *which* proxy moved, which is the only part an instructor can act on.
> - **Leakage is checked within a paper, not across the two.** A student sits one form; only what is printed on that form can help them. Cross-form resemblance is a *similarity* finding and is reported as its own indicator.
> - **The lenient net has three channels, unioned:** cosine ≥ 0.30, an *answer echo* (the other question states this one's answer — numbers and short answers matched whole), and shared rare vocabulary (≥5 terms and ≥20% of the combined vocabulary). The echo channel is the point: a question that states a constant in passing and a later one that asks for it embed far apart and leak completely.
> - **The budget is split between channels, not ranked across them.** A real 18-question paper has 153 pairs on it, so one report reads at most 16 — half the strongest echo pairs, half the closest pairs, unused share passed to the other. Ranking echo-first buried the closest pairs; ranking by similarity alone would never read the far-apart echo pair. There is no honest exchange rate between the two signals, so none is invented.
> - **Unread is not clean.** Pairs past the cap, and pairs whose verdict could not be parsed, are reported as *unchecked* — one collapsed note, not one per pair.
> - **A note that fires on everything points at nothing.** Per-question notes are collapsed once they stop discriminating: on a scanned course every question is OCR-sourced, so that becomes one note naming the count rather than twenty-seven identical sentences burying the leaks above them.

---

#### M11 · Instructor review UI + export
**Goal:** the decision surface, then the deliverable. (Brief Phases 9–10.)

Build:
- **Review screen** per question: editable stem, topic/type/level, score/time, system notes + source ref, and actions: edit, delete, approve, reject, regenerate, make easier/harder, clarify wording, move to other form. **Any instructor edit is saved and never auto-overwritten** by a later generation cycle.
- **Export** (MVP): exam PDF + answer-key PDF, with configurable options (form number, question order, institution logo, course data, duration, instructions, score distribution). Define the export interface so Word / Moodle XML / QTI / Canvas / Blackboard are later additions, not rewrites.

**Manual test:** edit a question, approve the set, export both PDFs, reopen — confirm your edit survived.
**Success check:** exported PDFs are correct and complete; instructor edits persist across regeneration cycles.

> **M11 technical decisions:**
> - **The lock is a flag, not an inference.** `Question.instructor_edited` is set at the one moment a human presses Save, and by nothing else — approving is a decision, not a rewrite, and does not set it. Inferring the lock from `updated_at` would have made every status change look like an edit.
> - **The automatic loop carries locked questions; it does not skip them.** The loop never rewrote a stored question, but it *replaces an item's attempt log* on every pass, and M9 reads the pool through that log — so an edited question whose stem no longer matched anything the model wrote would have silently dropped out of the pool. `locked_questions` are re-attached to the new log as round-0 attempts. An edit lost to bookkeeping is an edit lost.
> - **Manual request warns then obeys; the automatic cycle is silent.** `revise()` raises `RevisionNeedsConfirmation` *before* generating anything, so asking costs nothing. The loop says nothing at all, because the instructor did not ask for that run.
> - **Revisions travel M8's existing channel.** "Make easier" is a note handed to Agent 2A exactly the way Agent 3A's rejection notes are, and the replacement is reviewed by 3A before it is stored — a question an instructor asked for ends up on the same paper, so it meets the same bar. A rejected replacement leaves the original untouched. There is no second generation prompt.
> - **A replacement lands in the same `Question` row**, so the form placement, position and marks survive: Form A's question 7 stays Form A's question 7. It goes back to `candidate` and the lock comes off — approval was given to a different question.
> - **The exam document has nowhere to put an answer.** `build_exam_document` never reads `correct` or `answer_key`, so no exporter can leak one. Two files, always — one file with a section at the back is one careless print away from a lost exam.
> - **Export is a document model plus a renderer.** `export.py` decides what a paper *is*; `export_pdf.py` turns that into bytes. Word / Moodle XML / QTI / Canvas implement two methods and re-derive nothing.
> - **Arabic: wrap first, then shape, then reorder.** The bidi algorithm returns *visual* order, and visual-order text cannot be line-wrapped — so paragraphs are measured and broken while still logical, and each line is reshaped and reordered on its own. RTL papers also mirror their table columns and take their chrome vocabulary from the paper's language; per-question marks follow the *stem's* script, so an English question on an Arabic paper does not become one bidi-mixed line with back-to-front brackets.
> - **PDF via ReportLab + IBM Plex Sans Arabic (OFL, vendored).** Pure Python, no browser or system libraries, identical output on every machine that runs the suite; the font is vendored so an export works offline and on a machine with no Arabic system font. *(The plan's instruction was to follow `/mnt/skills/public/pdf/SKILL.md`; that skill is not present in this environment, so the stack was chosen deliberately and the seam above keeps swapping the renderer cheap.)*

---

#### M12 · Question bank
**Goal:** approved questions accumulate as a reusable asset. (Brief Phase 11.)

Build:
- `bank` app: on approval, instructor can save a question with stem+answer, topic, type, level, score, in-content source, creation date, and which exams used it.
- New-exam sourcing: fully new questions from content, ready questions from the bank (previously approved), or a mix at an instructor-set ratio.
- Bank search uses the same embeddings (`embed()`), so it moves to local with everything else.

**Manual test:** approve a question, save to bank, start a new exam, pull that question from the bank into a blueprint slot.
**Success check:** saved questions carry full metadata and are retrievable/reusable in a later exam for the same course.

> ✅ **MVP complete.** Every "ما يُبنى الآن" item from the brief's MVP scope is covered; every "ما يُؤجل" item is intentionally deferred behind a defined interface.

---

## 5. UI / UX design system

You said you need Django screens now but have no picture of them. Here is a complete, standards-following design system to hand to the agent. It is deliberately **not** one of the three overused AI-design defaults (cream + serif + terracotta / near-black + acid accent / broadsheet columns). The subject is an academic drafting tool used by faculty for focused review work, so the direction is **"quiet precision"**: calm, legible, document-like, with a single confident accent reserved for the one thing that matters most — the instructor's decision.

### Direction
An instructor uses this to *review and decide*, often for long stretches. So: low-glare surfaces, generous reading measure, no decoration competing with question text, and status/decision signaled by color used sparingly. The interface should feel like well-set paper with software affordances — not a dashboard.

### Color tokens
Ink-on-paper base with a deep teal as the single decision accent (approve/act), plus semantic status colors used only for status.

```css
:root {
  /* Surfaces — warm-neutral paper, not stark white */
  --surface-page:    #F7F6F3;   /* app background */
  --surface-card:    #FFFFFF;   /* question cards, panels */
  --surface-sunken:  #EFEDE8;   /* wells, table headers */

  /* Ink */
  --ink-900: #1C1D1B;   /* primary text */
  --ink-600: #4B4D48;   /* secondary text */
  --ink-400: #8A8C85;   /* captions, meta */
  --hairline: #DEDBD3;  /* hairline rules / borders */

  /* Accent — the decision color. Used for primary actions only. */
  --accent-700: #0E5E5A;   /* deep teal, primary button */
  --accent-500: #15807A;   /* hover */
  --accent-100: #E2F0EE;   /* accent tint background */

  /* Semantic status (used ONLY for status, never decoration) */
  --ok-600:    #2F7A43;   /* approved / passes check */
  --warn-600:  #A66A00;   /* needs attention */
  --danger-600:#B23A2E;   /* rejected / failed check */
  --info-600:  #2B5FA6;   /* system note */
}
```

### Typography
Two roles, chosen to read like a serious editorial/academic tool and to handle **Arabic + Latin + math notation** side by side (your content is bilingual).

- **Display / headings:** `"Fraunces"` (optical serif, characterful but restrained) for screen titles and section headers. Use at large sizes with tight leading.
- **Body / UI:** `"IBM Plex Sans"` and **`"IBM Plex Sans Arabic"`** — one family, two scripts, so Arabic and English UI text stay visually consistent. Excellent for dense forms and tables.
- **Data / monospace:** `"IBM Plex Mono"` for scores, timing, IDs, and formula-ish inline snippets.

```css
--font-display: "Fraunces", Georgia, serif;
--font-body:    "IBM Plex Sans", "IBM Plex Sans Arabic", system-ui, sans-serif;
--font-mono:    "IBM Plex Mono", ui-monospace, monospace;

/* Type scale (1.25 ratio) */
--fs-caption: 0.8rem;    /* meta, source refs */
--fs-body:    1rem;      /* 16px base */
--fs-lead:    1.25rem;   /* question stems — give them room */
--fs-h3:      1.563rem;
--fs-h2:      1.953rem;
--fs-h1:      2.441rem;

--lh-tight: 1.2;   /* headings */
--lh-read:  1.6;   /* question text, long content */
```
**Direction support:** set `dir="rtl"` on Arabic screens; the layout must mirror cleanly. Keep numbers, scores, and Latin technical terms LTR inside RTL text (`unicode-bidi: isolate`).

### Spacing, radius, elevation
```css
--space-1: 4px;  --space-2: 8px;  --space-3: 12px;
--space-4: 16px; --space-6: 24px; --space-8: 32px; --space-12: 48px;

--radius-sm: 6px;   --radius-md: 10px;   --radius-lg: 14px;
--shadow-card: 0 1px 2px rgba(28,29,27,.06), 0 2px 8px rgba(28,29,27,.04);
```
Keep elevation low — this is paper, not floating glass.

### Signature element
The **Decision Rail**: a slim vertical status strip on the left edge of every question card that encodes the question's state in one glance — grey (candidate), teal (approved), amber (needs attention), red (rejected). It's the one distinctive, functional device: color earns its place by carrying the exact information the instructor is here to act on. Everything else stays quiet.

### Core components (build once, reuse everywhere)
1. **Question card** — decision rail + editable stem (`--fs-lead`), meta row (topic · type · level · score · time in mono), collapsible system-notes panel (`--info-600`), source ref (caption), action bar.
2. **Action bar** — one **primary** button in `--accent-700` (the decision, e.g. *Approve*); everything else secondary/ghost. Never two primaries. Labels are verbs, consistent through the flow (a button that says *Approve* produces a toast that says *Approved*).
3. **Validation banner** — inline, states what's wrong and how to fix it, in the interface's voice, no apology. Empty states are invitations ("No topics yet — upload a file to begin").
4. **Comparison table** — sticky header, mono for numbers, status colors only on divergent cells, honest labels ("expected time," "expected difficulty").
5. **Blueprint editor** — editable table; live validation total row that turns `--danger-600` when weights don't sum.
6. **Stepper** — the instructor journey (Course → Upload → Topics → Spec → Blueprint → Generate → Review → Forms → Compare → Approve → Export), current step in accent, done steps in ok.

### Quality floor (non-negotiable, per UI standards)
- Responsive to mobile (cards stack; tables scroll horizontally with a sticky first column).
- Visible keyboard focus on every interactive element (`:focus-visible` ring in `--accent-500`).
- `prefers-reduced-motion` respected — transitions ≤150ms, none essential.
- Color never the sole signal: pair every status color with an icon or text label (accessibility + color-blind safety).
- Contrast ≥ WCAG AA for all text.

### Screen inventory (maps 1:1 to milestones)
| Screen | Milestone | Primary job |
|--------|-----------|-------------|
| Courses dashboard | M0/M1 | list + create courses |
| Course detail / upload | M1 | upload files, see extraction |
| Topic review | M2 | edit/merge/exclude topics |
| Exam spec | M4 | set forms, score, count, duration, language, weights |
| Blueprint editor | M4 | adjust rows, live validation |
| Generation run | M8 | watch the agent loop, see candidates |
| Question review | M11 | the decision surface (signature card) |
| Forms & compare | M10 | side-by-side, honest indicators |
| Export | M11 | configure + download PDFs |
| Question bank | M12 | search, reuse |

---

## 6. Validation plan (run alongside the build)

Use the **three sample courses** — one theoretical, one math-heavy, one programming — as your standing test bed from M2 onward.

- **Extraction accuracy** (M2): do extracted topics match the real sections?
- **Scope adherence** (M5): pick one chapter, confirm every generated question stays inside it.
- **Level match** (M7): request direct / medium / multi-step levels, judge adherence yourself.
- **Form convergence + MCQ quality** (M10): compare two forms on coverage/type/score/time; check distractors, ambiguity, multiple-answer bugs.
- **Pilot** (after M12): a small group of faculty each build one real or trial exam and rate the experience — ease of use, question quality, content fit, usefulness of controls, share of questions accepted directly, time vs their usual method. **No student scores needed** at this stage; the goal is drafting + review help before deployment.

## 7. Success metrics (track from first real use)
1. Time to create an exam (down).
2. Share of questions accepted with no edit.
3. Share needing substantial edit.
4. Accuracy of adherence to selected topics.
5. Count of problems the system caught before approval.
6. Convergence of resulting forms.
7. Reduction in edits needed after form-building.
8. Faculty satisfaction.

---

## 8. Local-LLM switch (Stage 5, after things are stable)

Once the API-key version is proven:
1. Write `AirLLMProvider(LLMProvider)` implementing `complete()` and `embed()` against your AirLLM-served local model.
2. Flip `LLM_PROVIDER=airllm` in `.env`. At the same time, **PaddleOCR-VL replaces Gemini OCR** — flip `OCR_PROVIDER=paddlevl` so ingestion goes local too.
3. Re-run the **M2, M5, M7, M10** validation checks on the three sample courses to confirm quality holds locally.
4. No other code changes — that's the whole point of §3.

---

## 9. Order of operations checklist (hand to the agent)

```
[ ] §3  LLMProvider + llm_ping        (before anything)
[ ] M0  skeleton + auth + design system
[ ] M1  course + upload + extraction
[ ] M2  topics + topic review  + chunking/embeddings
[ ] M3  retrieval (RAG)                ── Agent 1A ──
[ ] M4  blueprint + validation         ── Agent 1A done ──
[ ] M5  question generation            ── Agent 2A ──
[ ] M6  answer key with question       ── Agent 2A done ──
[ ] M7  review checks                  ── Agent 3A ──
[ ] M8  closed correction loop         ── Agent 3A done ──
[ ] M9  form assembly A & B
[ ] M10 convergence + compare screen
[ ] M11 review UI + PDF export
[ ] M12 question bank
[ ] S5  AirLLM local provider + re-validate
```

Build in this order. Do not skip a milestone's success check. The moment a stage's agent passes its check, you have something real to try — which is exactly how this plan is meant to be used.


