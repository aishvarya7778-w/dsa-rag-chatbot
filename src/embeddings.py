"""
Embeddings module for the DSA RAG Chatbot.
Wraps LangChain's Google GenAI embeddings using gemini-embedding-001
with explicit output_dimensionality=768 and robust exponential backoff
for HTTP 429 / RESOURCE_EXHAUSTED quota errors.
"""

import re
import time
import random
from typing import Optional, List, Any, Callable, Dict
import config

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None


class QuotaExhaustedError(Exception):
    """
    Raised when Gemini embedding API quota is exhausted and retries have failed,
    or when a hard quota limit has been encountered.
    """

    def __init__(
        self,
        message: str,
        retry_delay: Optional[float] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.retry_delay = retry_delay
        self.original_exception = original_exception


def extract_retry_delay(exc: Exception) -> Optional[float]:
    """
    Extract suggested retry delay in seconds from a Gemini API exception if available.
    Inspects structured details dicts and error message strings.
    """
    # 1. Check structured details on the exception or its cause
    for target in (exc, getattr(exc, "__cause__", None)):
        if target is None:
            continue
        details = getattr(target, "details", None)
        if isinstance(details, dict):
            err_dict = details.get("error", details) if isinstance(details, dict) else {}
            item_list = err_dict.get("details", []) if isinstance(err_dict, dict) else []
            if isinstance(item_list, list):
                for item in item_list:
                    if isinstance(item, dict) and "retryDelay" in item:
                        delay_str = str(item["retryDelay"])
                        try:
                            return float(delay_str.rstrip("s"))
                        except ValueError:
                            pass

    # 2. Regex fallback on error string representation
    exc_str = str(exc)
    # Match patterns like: "retry in 34.2s", "retry after 34 seconds", "retryDelay: '34s'"
    m = re.search(r"retry\s+(?:in|after)\s+([0-9.]+)\s*s?", exc_str, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"['\"]?retryDelay['\"]?\s*:\s*['\"]?([0-9.]+)\s*s?['\"]?", exc_str, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return None


def is_quota_or_rate_limit_error(exc: Exception) -> bool:
    """
    Detect if an exception is an HTTP 429 / RESOURCE_EXHAUSTED / quota limit error.
    """
    targets = [exc]
    if getattr(exc, "__cause__", None):
        targets.append(exc.__cause__)

    for target in targets:
        code = getattr(target, "code", getattr(target, "status_code", None))
        if code == 429:
            return True
        status = getattr(target, "status", None)
        if status == "RESOURCE_EXHAUSTED":
            return True
        target_str = str(target)
        if "429" in target_str and ("RESOURCE_EXHAUSTED" in target_str or "quota" in target_str.lower()):
            return True
        if "quota exceeded" in target_str.lower() or "resource_exhausted" in target_str.lower():
            return True

    return False


def embed_with_backoff(
    embed_fn: Callable[[], Any],
    operation_name: str = "embeddings",
    max_retries: int = config.MAX_RETRIES,
    initial_delay: float = config.INITIAL_RETRY_DELAY,
    backoff_factor: float = config.RETRY_BACKOFF_FACTOR,
) -> Any:
    """
    Execute an embedding function with exponential backoff on HTTP 429 / quota errors.
    Respects API suggested retry delays when available.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return embed_fn()
        except Exception as exc:
            if not is_quota_or_rate_limit_error(exc):
                # Non-quota error; do not backoff, raise immediately
                raise exc

            suggested_delay = extract_retry_delay(exc)

            if attempt >= max_retries:
                raise QuotaExhaustedError(
                    f"Gemini API quota remains exhausted after {max_retries} retries during {operation_name}.",
                    retry_delay=suggested_delay,
                    original_exception=exc,
                ) from exc

            # Calculate wait time: respect API suggested delay if present, else exponential backoff
            if suggested_delay is not None:
                wait_time = suggested_delay + 1.5
            else:
                jitter = random.uniform(0.5, 1.5)
                wait_time = (initial_delay * (backoff_factor ** (attempt - 1))) + jitter

            print(
                f"[!] Rate limit (HTTP 429) during {operation_name}. "
                f"Waiting {wait_time:.1f}s before retry (attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait_time)


if GoogleGenerativeAIEmbeddings is not None:
    class ResilientGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
        """
        Subclass of GoogleGenerativeAIEmbeddings with exponential backoff and
        automatic retry delay respect for HTTP 429 / RESOURCE_EXHAUSTED quota limits.
        """
        max_retries: int = config.MAX_RETRIES
        initial_delay: float = config.INITIAL_RETRY_DELAY
        backoff_factor: float = config.RETRY_BACKOFF_FACTOR

        def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return embed_with_backoff(
                lambda: super(ResilientGoogleGenerativeAIEmbeddings, self).embed_documents(texts, **kwargs),
                operation_name=f"batch of {len(texts)} chunks",
                max_retries=self.max_retries,
                initial_delay=self.initial_delay,
                backoff_factor=self.backoff_factor,
            )

        def embed_query(self, text: str, **kwargs: Any) -> List[float]:
            return embed_with_backoff(
                lambda: super(ResilientGoogleGenerativeAIEmbeddings, self).embed_query(text, **kwargs),
                operation_name="query embedding",
                max_retries=self.max_retries,
                initial_delay=self.initial_delay,
                backoff_factor=self.backoff_factor,
            )
else:
    ResilientGoogleGenerativeAIEmbeddings = None


def get_embedding_model(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    output_dimensionality: Optional[int] = None,
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
) -> "GoogleGenerativeAIEmbeddings":
    """
    Instantiate and return the Google Generative AI Embeddings model with retry backoff.
    
    Args:
        api_key: Optional Gemini API key. Defaults to config.GEMINI_API_KEY.
        model_name: Optional embedding model name. Defaults to config.EMBEDDING_MODEL (gemini-embedding-001).
        output_dimensionality: Optional output vector dimension. Defaults to config.EMBEDDING_DIMENSION (768).
        max_retries: Optional retry count on 429. Defaults to config.MAX_RETRIES.
        initial_delay: Optional initial delay seconds. Defaults to config.INITIAL_RETRY_DELAY.
        backoff_factor: Optional multiplier for backoff. Defaults to config.RETRY_BACKOFF_FACTOR.
        
    Returns:
        Configured ResilientGoogleGenerativeAIEmbeddings instance with 768 dimensions.
        
    Raises:
        ImportError: If langchain-google-genai is not installed.
        ValueError: If GEMINI_API_KEY is missing.
    """
    if GoogleGenerativeAIEmbeddings is None:
        raise ImportError(
            "langchain-google-genai is not installed. Run `pip install langchain-google-genai`."
        )

    resolved_key = config.GEMINI_API_KEY if api_key is None else api_key
    if not resolved_key:
        raise ValueError(
            "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
        )

    resolved_model = model_name or config.EMBEDDING_MODEL
    resolved_dim = (
        output_dimensionality
        if output_dimensionality is not None
        else config.EMBEDDING_DIMENSION
    )
    resolved_max_retries = max_retries if max_retries is not None else config.MAX_RETRIES
    resolved_initial_delay = initial_delay if initial_delay is not None else config.INITIAL_RETRY_DELAY
    resolved_backoff = backoff_factor if backoff_factor is not None else config.RETRY_BACKOFF_FACTOR

    return ResilientGoogleGenerativeAIEmbeddings(
        model=resolved_model,
        google_api_key=resolved_key,
        output_dimensionality=resolved_dim,
        max_retries=resolved_max_retries,
        initial_delay=resolved_initial_delay,
        backoff_factor=resolved_backoff,
    )
