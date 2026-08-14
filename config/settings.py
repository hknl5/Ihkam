"""
Django settings for the إحكام (Ihkam) project.

Secrets and environment-specific values are read from a local `.env` file
(see `.env.example`). Nothing secret belongs in this file.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Core -------------------------------------------------------------------

SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-insecure-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [h.strip() for h in env("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "accounts",
    "agents",
    "courses",
    "exams",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# --- Database ---------------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("POSTGRES_DB", "ihkam"),
        "USER": env("POSTGRES_USER", "postgres"),
        "PASSWORD": env("POSTGRES_PASSWORD", ""),
        "HOST": env("POSTGRES_HOST", "127.0.0.1"),
        "PORT": env("POSTGRES_PORT", "5432"),
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# --- i18n / l10n ------------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = env("DJANGO_TIME_ZONE", "Asia/Riyadh")
USE_I18N = True
USE_TZ = True

# --- Static -----------------------------------------------------------------

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

# --- Media (uploaded course material) ---------------------------------------

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / env("DJANGO_MEDIA_ROOT", "media")

# Lecture decks are large; keep big uploads on disk rather than in memory.
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024

# --- Auth flow --------------------------------------------------------------

LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "courses:dashboard"
LOGOUT_REDIRECT_URL = "accounts:login"

# --- LLM provider (see agents/provider.py, §3 of the build plan) -------------

LLM_PROVIDER = env("LLM_PROVIDER", "openai").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = env("OPENAI_EMBED_MODEL", "text-embedding-3-small")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = env("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Local model, phase 2 (see §8). Not implemented yet.
AIRLLM_MODEL = os.getenv("AIRLLM_MODEL", "")
AIRLLM_EMBED_MODEL = os.getenv("AIRLLM_EMBED_MODEL", "")

# --- OCR provider (see agents/ocr.py) ---------------------------------------
# Image-only pages are read by a vision-language model, not a classic OCR
# engine: the material is Arabic + English and VLMs read Arabic far better.
# gemini | paddlevl (paddlevl is the local phase-2 stub, not implemented yet)

OCR_PROVIDER = env("OCR_PROVIDER", "gemini").strip().lower()

# Reuses the Gemini key already configured above. Defaults to the same model
# as LLM work, which is vision-capable.
GEMINI_OCR_MODEL = os.getenv("GEMINI_OCR_MODEL", "").strip() or GEMINI_MODEL

# Pages are rasterised at this width (px) before being sent for OCR. Wide
# enough for small diagram labels, small enough to keep the request cheap.
OCR_RENDER_WIDTH = int(env("OCR_RENDER_WIDTH", "1600"))

# OCR runs during upload. Turn it off to upload without spending API calls
# (tests do this); pages then simply stay flagged as needing OCR.
OCR_ENABLED = env_bool("OCR_ENABLED", True)

# One image per call — batching pages into a single request measurably
# degrades transcription quality. Concurrency shortens the wait instead.
OCR_CONCURRENCY = int(env("OCR_CONCURRENCY", "5"))

# Upper bound on pages OCR'd per file. A file over the cap is not rejected:
# the first pages are read and the rest stay flagged, with a clear message.
OCR_MAX_PAGES_PER_FILE = int(env("OCR_MAX_PAGES_PER_FILE", "60"))

# Attempts per page when the provider rate-limits us. Free Gemini keys allow
# only a few requests per minute, so a page is retried rather than lost.
OCR_MAX_ATTEMPTS = int(env("OCR_MAX_ATTEMPTS", "5"))

# --- When a page is re-read by OCR (see courses/services/ingest.py) ---------
# A little extractable text is not proof a page is complete. Three shapes of
# incomplete page are re-read; every threshold below was measured on the real
# uploaded lecture files, and the numbers are recorded next to each one so a
# later change can be argued against the same evidence.

# A "mixed" page: a real text layer for the title, with the body sitting in an
# image. Both conditions must hold — either alone misfires on real pages.
# Measured on ch10.3.pdf: the broken pages (3, 4, 5, 7, 35) have 14-36 letters
# under 0.27-0.41 image coverage, while complete pages carry 104-366 letters
# (page 31 is complete under 0.49 coverage, so coverage alone would re-read it,
# and page 37 has 47 letters with no image at all, so letters alone would too).
OCR_MIXED_MAX_LETTERS = int(env("OCR_MIXED_MAX_LETTERS", "80"))
OCR_MIXED_MIN_IMAGE_COVERAGE = float(env("OCR_MIXED_MIN_IMAGE_COVERAGE", "0.20"))

# A "defective font" page: a full text layer whose embedded fonts have broken
# ToUnicode tables, so letters arrive as unmappable junk. Measured on the
# Arabic guidelines file, where the junk sits in the headings: per-page counts
# run 0, 0, 1, 2, 4, then 8, 9, 12 ... 39. The gap at 4 → 8 is the widest in
# that range, which is where the floor goes — one or two stray glyphs are not
# worth an OCR call, a mangled heading is.
OCR_DEFECTIVE_MIN_CHARS = int(env("OCR_DEFECTIVE_MIN_CHARS", "8"))
# A floor, not the main test: it only stops a handful of stray glyphs on a very
# long page from counting. Ratio cannot lead, because on a short page 2 junk
# chars already reach 0.03.
OCR_DEFECTIVE_MIN_RATIO = float(env("OCR_DEFECTIVE_MIN_RATIO", "0.005"))

# A transcription that comes back far shorter than the text layer it would
# replace is treated as truncated, and the text layer is kept. This only ever
# guards pages that already had substantial text (the defective-font case);
# an image-only or mixed page has almost none to lose.
OCR_MIN_KEEP_RATIO = float(env("OCR_MIN_KEEP_RATIO", "0.6"))

# Embedding dimension the app stores in pgvector. Providers are asked to emit
# this width so switching providers does not invalidate stored vectors.
#
# Changing it after the first migration is a schema change, not a config
# change: `Chunk.embedding` is a fixed-width column, so a new value needs a
# migration and a re-embed of everything already stored.
EMBEDDING_DIM = int(env("EMBEDDING_DIM", "1536"))

# --- Chunking + embeddings (M2, see courses/services/chunking.py) -----------

# Chunking runs at the end of ingest, so a file is retrievable as soon as it is
# readable. Turn it off to upload without spending embedding calls (the test
# suite does this); pages then simply have no chunks until they are rebuilt.
EMBEDDINGS_ENABLED = env_bool("EMBEDDINGS_ENABLED", True)

# A passage never crosses a page boundary — a citation is only as good as the
# page number on the passage it came from. Within a page, paragraphs are packed
# up to the maximum; a paragraph on its own below the minimum is joined to the
# next rather than becoming a chunk of one line. Slide decks mostly produce one
# chunk per page at these sizes; dense document pages produce two or three.
CHUNK_MAX_CHARS = int(env("CHUNK_MAX_CHARS", "1200"))
CHUNK_MIN_CHARS = int(env("CHUNK_MIN_CHARS", "200"))

# Passages per embedding request. Only affects how many round trips a file
# costs, never what is stored.
EMBED_BATCH_SIZE = int(env("EMBED_BATCH_SIZE", "64"))

# --- Retrieval (M3, see courses/services/retrieval.py) ----------------------

# How many passages one query may return, before the score floor is applied.
# A blueprint row needs enough material to ground a question in, not a reading
# list: 8 passages is roughly two pages of a slide deck.
RETRIEVAL_TOP_K = int(env("RETRIEVAL_TOP_K", "8"))

# Cosine similarity floor. Below it a passage is not a worse answer to the
# query, it is a different subject, and padding the result out to TOP_K with
# those is how unrelated chapters leak into a generated question.
#
# Measured over 12 queries on the three sample courses (discrete maths, the
# Arabic guidelines, the OOP slides) with gemini-embedding-001 at 1536
# dimensions. The absolute numbers are higher than they look: every passage in
# one lecture file shares its subject, so nothing scores near zero. On-topic
# passages land at 0.65-0.81; the first passage from a different chapter lands
# at 0.55-0.65. 0.65 is the boundary between them.
#
# It is deliberately on the strict side of that boundary. A borderline relevant
# passage is occasionally cut (a "constructor declaration" page at 0.647 on an
# overloading query), which costs one reference; an unrelated chapter admitted
# instead would ground a generated question in material the topic does not
# cover, which is the failure M3 exists to prevent.
#
# It is a per-corpus number, not a universal one — re-measure it if the
# embedding model or EMBEDDING_DIM changes.
RETRIEVAL_MIN_SCORE = float(env("RETRIEVAL_MIN_SCORE", "0.65"))

# --- Topic extraction (M2, see courses/services/topics.py) ------------------

# Ceiling on the course text sent in one extraction call. A file over the
# ceiling is not silently truncated: the pages that did not fit are named in
# the result, so the instructor knows which part of the syllabus was read.
TOPIC_EXTRACTION_MAX_CHARS = int(env("TOPIC_EXTRACTION_MAX_CHARS", "150000"))
