# إحكام (Ihkam)

Intelligent agentic system for exam generation and review.
إحكام prepares the draft and surfaces imbalances. **The final decision always belongs to the instructor.**

Build status: **M0 complete** — project skeleton, instructor auth, the `LLMProvider`
abstraction (§3), and the design system (§5). Nothing from M1 onward exists yet.

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

## Tests

```bash
uv run python manage.py test          # everything (needs PostgreSQL)
uv run python manage.py test tests.test_provider   # no DB, no network
```
