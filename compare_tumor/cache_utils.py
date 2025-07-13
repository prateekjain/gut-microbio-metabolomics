# cache_utils.py
import time
import json
import hashlib
from pathlib import Path
import pandas as pd
import joblib  # pip install joblib
import functools
import logging

_CACHE_DIR = Path(__file__).parent / "__page_cache__"
_MEMORY = joblib.Memory(_CACHE_DIR, verbose=0)  # Filesystem backend
_DEFAULT_TTL = 900  # 15 min
_DEFAULT_MAX_SIZE = 50
_METADATA_FILE = _CACHE_DIR / "cache_metadata.json"

logging.info(f"Disk cache initialized at: {_CACHE_DIR}")
_CACHE_DIR.mkdir(exist_ok=True)

def _load_metadata():
    """Loads cache metadata from a JSON file, returning empty dict on error."""
    if not _METADATA_FILE.exists():
        return {}
    try:
        with open(_METADATA_FILE, 'r') as f:
            # For concurrent access, a file lock would be needed here
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # On error, treat cache as empty and proceed
        return {}

def _save_metadata(metadata):
    """Saves cache metadata to a JSON file."""
    try:
        with open(_METADATA_FILE, 'w') as f:
            # For concurrent access, a file lock would be needed here
            json.dump(metadata, f, indent=4)
    except IOError as e:
        logging.error(f"Could not write cache metadata: {e}")

def _normalise(obj):
    """
    Produce a deterministic, hashable representation for cache keys.
      - Sorts lists, tuples, and sets.
      - Recursively processes nested structures.
      - Converts dictionaries to sorted tuple of items.
    """
    if isinstance(obj, (list, tuple, set)):
        return tuple(sorted(_normalise(x) for x in obj))
    if isinstance(obj, dict):
        return tuple(sorted((k, _normalise(v)) for k, v in obj.items()))
    return obj

def _make_key(func, *args, **kwargs):
    """Creates a stable SHA1 hash for the function call."""
    # Include the module and function name for uniqueness
    func_identifier = f"{func.__module__}.{func.__name__}"
    
    # Normalize all arguments to ensure cache hits are consistent
    # e.g., func(a=1, b=2) and func(b=2, a=1) produce the same key
    normalized_args = _normalise(args)
    normalized_kwargs = _normalise(kwargs)
    
    payload = (func_identifier, normalized_args, normalized_kwargs)
    
    # Use json.dumps with a default handler for un-serializable types
    # like dataframes, which will be handled by their string representation.
    payload_str = json.dumps(payload, default=str, sort_keys=True)
    
    return hashlib.sha1(payload_str.encode()).hexdigest()

def page_cache(ttl=_DEFAULT_TTL):
    """
    A persistent, argument-aware cache decorator.
    
    Features:
      - Caches results to disk, surviving process restarts.
      - Normalizes list/dict arguments, so argument order doesn't matter.
      - Uses a time-to-live (TTL) to invalidate stale data.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = _make_key(func, *args, **kwargs)
            cache_path = _MEMORY.store_backend.get_path((key, None))
            
            try:
                # Manually check TTL to avoid reading stale files
                if cache_path.exists():
                    timestamp = cache_path.stat().st_mtime
                    if (time.time() - timestamp) < ttl:
                        logging.debug(f"Cache hit for {func.__name__} (key: {key[:7]}...)")
                        return _MEMORY.store_backend.load_item((key, None))
                    else:
                        logging.debug(f"Cache expired for {func.__name__} (key: {key[:7]}...)")
            except FileNotFoundError:
                pass  # Cache miss
            except Exception as e:
                logging.warning(f"Cache read error for {func.__name__}: {e}. Recalculating.")

            # Cache miss or expired, execute the function
            logging.debug(f"Cache miss for {func.__name__} (key: {key[:7]}...). Executing function.")
            result = func(*args, **kwargs)
            
            # Store the fresh result
            try:
                _MEMORY.store_backend.dump_item((key, None), result)
            except Exception as e:
                logging.error(f"Failed to write to cache for {func.__name__}: {e}")
                
            return result
        return wrapper
    return decorator 