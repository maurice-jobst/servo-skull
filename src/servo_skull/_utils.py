"""Shared utilities for servo-skull."""
import hashlib
import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


def calculate_checksum(data: bytes) -> str:
    """Calculate SHA256 checksum of data."""
    return hashlib.sha256(data).hexdigest()


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(f"{func.__name__} attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            raise RuntimeError(f"Retries exhausted for {func.__name__}")
        return wrapper
    return decorator


def safe_json_read(path: Path) -> dict[str, Any] | None:
    """Safely read JSON file, return None if invalid."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read JSON from {path}: {e}")
        return None


def safe_json_write(path: Path, data: dict[str, Any]) -> bool:
    """Safely write JSON file, return True if successful."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        return True
    except (OSError, TypeError) as e:
        logger.error(f"Failed to write JSON to {path}: {e}")
        return False


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logger with console handler (prevents duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def bootstrap_env():
    """Bootstrap environment variables from standard .env file locations."""
    import os
    from pathlib import Path
    
    home = Path.home()
    paths = [
        Path.cwd() / ".env",
        home / ".config" / "tech-priest" / ".env",
        Path(__file__).parent.parent.parent.parent / ".env",
    ]
    
    # Try importing python-dotenv first
    try:
        from dotenv import load_dotenv
        for p in paths:
            if p.exists():
                load_dotenv(p)
                return
    except ImportError:
        pass
        
    # Manual fallback parser in case python-dotenv is missing
    for p in paths:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
                return
            except Exception:
                pass


# Bootstrap environment upon importing utils
bootstrap_env()


def count_syllables(word: str) -> int:
    """Estimate the number of syllables in a word."""
    word = word.lower().strip()
    if not word:
        return 0
    # Simple check for short words
    if len(word) <= 3:
        return 1
    
    orig_word = word
    # Remove some common suffixes
    if word.endswith(("es", "ed")):
        # exception: words ending with le (e.g. table)
        if not word.endswith("le"):
            word = word[:-2]
    elif word.endswith("e"):
        # silent e: strip it
        word = word[:-1]
            
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    in_vowel_group = False
    for char in word:
        if char in vowels:
            if not in_vowel_group:
                count += 1
                in_vowel_group = True
        else:
            in_vowel_group = False
            
    # Adjust count for some common patterns
    if orig_word.endswith("le") and len(orig_word) > 2 and orig_word[-3] not in vowels:
        count += 1
        
    return max(1, count)


def calculate_flesch_reading_ease(text: str) -> float:
    """
    Calculate the Flesch Reading Ease score of the text.
    Excludes code blocks, HTML, and technical acronyms/system terms.
    """
    import re
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Exclude system terminology and uppercase acronyms (like SKU100, API, SDK, HTTP, JSON, OAUTH)
    # Words with digits (e.g. SKU100, E1, v6.0) or all-caps longer than 1 character
    words_raw = re.findall(r'\b[a-zA-Z]+\b', text)
    
    # Keep words that are not all uppercase (unless they are short, but let's exclude all-caps acronyms > 1 char)
    words = []
    for w in words_raw:
        if w.isupper() and len(w) > 1:
            # Likely an acronym (e.g., API, SDK, HTTP, JSON, OAUTH)
            continue
        words.append(w)
        
    if not words:
        return 0.0
        
    # Count sentences (split on ., !, ?, but watch for abbreviations)
    # Simple sentence count: split by . ! ? that are followed by space or end of string
    sentences = re.split(r'[.!?]+(?:\s+|$)', text.strip())
    sentences = [s for s in sentences if s.strip()]
    num_sentences = max(1, len(sentences))
    
    num_words = len(words)
    
    # Count syllables
    num_syllables = sum(count_syllables(w) for w in words)
    
    # Flesch Reading Ease formula
    # 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return round(score, 2)


def _load_config() -> dict[str, Any]:
    """Load configuration from providers.toml."""
    import tomllib
    paths_to_try = [
        Path(__file__).parent.parent.parent.parent.parent / "config" / "providers.toml",
        Path(__file__).parent.parent.parent.parent / "config" / "providers.toml",
        Path(__file__).parent.parent.parent / "config" / "providers.toml",
    ]
    
    config_path = None
    for p in paths_to_try:
        if p.exists():
            config_path = p
            break
            
    if config_path is None:
        logger.warning("Config file providers.toml not found, using defaults")
        return {
            "providers": {
                "local_gemma": {
                    "type": "ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "model": "gemma4:26b",
                    "temperature": 0.0,
                    "timeout": 300.0,
                },
                "cloud_openai": {
                    "type": "openai",
                    "api_key": "",
                    "model": "gpt-4",
                    "timeout": 300.0,
                }
            },
        }

    with open(config_path, "rb") as f:
        return tomllib.load(f)



