"""Identity helpers — distinct_id hashing.

`distinct_id` is the per-user key we send to PostHog. By default it's a
SHA-256 of `"{channel}:{identifier}"` — stable and PII-free, but
channel-scoped: the same human across WhatsApp, email, and Telegram gets
three distinct_ids and looks like three members of the household.

To collapse those back to one human, homer writes an identity map
(JSON, flat `"channel:identifier"` → `"person:<slug>"`) and points
`HOMER_IDENTITY_MAP` at it. When a lookup hits, we hash the canonical
`person:<slug>` key instead, so every channel for that human produces
the same distinct_id. Unmapped senders fall through to channel-scoped
hashing — behaviour identical to pre-map deployments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def get_distinct_id(identifier: str, channel: str) -> str:
    """Return a stable SHA-256 hash for a user identity.

    If the identity map canonicalizes this (channel, identifier) pair,
    the canonical person key is hashed. Otherwise falls back to a
    channel-scoped hash.
    """
    canonical = _resolve_canonical(channel, identifier)
    key = canonical if canonical else f"{channel}:{identifier.strip()}"
    return _hash_identity_key(key)


def get_household_id() -> str:
    """Return the household UUID from HOMER_HOUSEHOLD_ID env var."""
    return os.environ.get("HOMER_HOUSEHOLD_ID", "")


# ── canonicalization ──────────────────────────────────────────────────────


def _hash_identity_key(key: str) -> str:
    """Hash a normalized identity key. Shared by get_distinct_id and the
    seen_users migration code in hook.py so they always agree."""
    return hashlib.sha256(key.lower().strip().encode()).hexdigest()


def _resolve_canonical(channel: str, identifier: str) -> str | None:
    """Look up (channel, identifier) in the configured identity map.

    Returns the canonical person key (e.g. `"person:ebby"`) on hit,
    None on miss or when no map is configured.
    """
    mapping = _load_identity_map()
    if not mapping:
        return None
    key = f"{channel}:{identifier.strip()}".lower()
    return mapping.get(key)


def _load_identity_map() -> dict[str, str]:
    """Load the identity map from HOMER_IDENTITY_MAP, cached by file mtime.

    Returning {} on any failure means callers transparently fall through
    to channel-scoped hashing — the pre-map behaviour — so a missing
    or broken map never breaks event capture.
    """
    path = os.environ.get("HOMER_IDENTITY_MAP", "").strip()
    if not path:
        return {}
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return {}
    return _load_identity_map_cached(path, mtime)


@lru_cache(maxsize=4)
def _load_identity_map_cached(path: str, mtime: float) -> dict[str, str]:
    # mtime is in the cache key so a rewrite invalidates automatically —
    # homer can rebuild the map without nanobot needing a restart.
    try:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("identity map is not a JSON object")
        return {
            str(k).lower().strip(): str(v).lower().strip()
            for k, v in data.items()
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load identity map at %s: %s", path, exc)
        return {}


def iter_identity_map() -> list[tuple[str, str]]:
    """Expose the loaded identity map as a list of (channel_key, person_key)
    pairs — used by hook.py to migrate existing seen_users entries after
    the map is first populated."""
    return list(_load_identity_map().items())
