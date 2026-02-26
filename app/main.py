"""
FastAPI application — Multilingual NMT Translation Service.

Endpoints
---------
POST /translate          Translate text between any supported language pair.
GET  /languages          List all supported languages from languages.yaml.
GET  /models             List all available trained model checkpoints.
GET  /health             Health-check / readiness probe.

Run
---
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Or from project root:
    python -m uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
import sys
import yaml

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference.translator import Translator, ModelNotFoundError


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multilingual NMT API",
    description=(
        "Neural Machine Translation service powered by a Seq2Seq model with "
        "Bahdanau Attention.  Supports English↔Hindi and is designed to extend "
        "to any Indian regional language without code changes."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialised translator singleton
_translator: Translator | None = None


def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator(max_output_len=100)
    return _translator


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        example="Hello, how are you?",
        description="Input sentence to translate.",
    )
    source_lang: str = Field(
        ...,
        min_length=2,
        max_length=5,
        example="en",
        description="ISO-639-1 source language code.",
    )
    target_lang: str = Field(
        ...,
        min_length=2,
        max_length=5,
        example="hi",
        description="ISO-639-1 target language code.",
    )

    @validator("source_lang", "target_lang")
    def lowercase_lang(cls, v):
        return v.strip().lower()


class TranslateResponse(BaseModel):
    translation : str
    source_lang : str
    target_lang : str


class LanguageInfo(BaseModel):
    code   : str
    name   : str
    script : str


class ModelInfo(BaseModel):
    source_lang : str
    target_lang : str
    checkpoint  : str


# ---------------------------------------------------------------------------
# Load supported languages once at startup
# ---------------------------------------------------------------------------
def _load_supported_languages() -> dict:
    lang_file = os.path.join(PROJECT_ROOT, "config", "languages.yaml")
    if os.path.exists(lang_file):
        with open(lang_file, encoding="utf-8") as f:
            return yaml.safe_load(f).get("supported_languages", {})
    return {}


SUPPORTED_LANGUAGES: dict = _load_supported_languages()


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "detail": f"Model for {exc.src_lang}→{exc.tgt_lang} not found. "
                      f"Train it first using training/train.py."
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health():
    """Readiness / liveness probe."""
    return {"status": "ok", "service": "Multilingual NMT API"}


@app.get("/languages", response_model=list[LanguageInfo], tags=["Info"])
async def list_languages():
    """
    Return the list of all languages supported by the system config.

    Note: Presence here does not guarantee a trained model is available.
    Use /models to see which pairs have checkpoints.
    """
    return [
        LanguageInfo(
            code=code,
            name=info.get("name", code),
            script=info.get("script", ""),
        )
        for code, info in SUPPORTED_LANGUAGES.items()
    ]


@app.get("/models", response_model=list[ModelInfo], tags=["Info"])
async def list_models():
    """
    Scan the saved_models/ directory and return all available language pairs
    with trained checkpoints.
    """
    translator = get_translator()
    models_dir = os.path.join(PROJECT_ROOT, "saved_models")
    pairs = translator.available_pairs()
    return [
        ModelInfo(
            source_lang=src,
            target_lang=tgt,
            checkpoint=os.path.join(models_dir, f"model_{src}_{tgt}.pt"),
        )
        for src, tgt in pairs
    ]


@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate(request: TranslateRequest):
    """
    Translate text from one language to another.

    **Request body**
    ```json
    {
      "text": "Hello world",
      "source_lang": "en",
      "target_lang": "hi"
    }
    ```

    **Response**
    ```json
    {
      "translation": "नमस्ते दुनिया",
      "source_lang": "en",
      "target_lang": "hi"
    }
    ```

    **Error responses**
    - `404` — Model for the requested language pair has not been trained yet.
    - `422` — Input validation error (empty text, unsupported lang code, etc.).
    """
    # Validate language codes against config
    if SUPPORTED_LANGUAGES and request.source_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Source language '{request.source_lang}' is not in supported languages: "
                   f"{list(SUPPORTED_LANGUAGES.keys())}",
        )
    if SUPPORTED_LANGUAGES and request.target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Target language '{request.target_lang}' is not in supported languages: "
                   f"{list(SUPPORTED_LANGUAGES.keys())}",
        )

    translator = get_translator()
    result = translator.translate(
        request.text, request.source_lang, request.target_lang
    )

    return TranslateResponse(
        translation=result.translation,
        source_lang=result.src_lang,
        target_lang=result.tgt_lang,
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
