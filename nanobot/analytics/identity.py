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

# Shared no-op mapping returned on the unconfigured-map path. Reusing one
# instance avoids a dict allocation on every inbound message when
# HOMER_IDENTITY_MAP is unset (the common case outside hosted-homer).
_EMPTY_MAP: dict[str, str] = {}


def get_distinct_id(identifier: str, channel: str) -> str:
    """Return a stable SHA-256 hash for a user identity.

    If the identity map canonicalizes this (channel, identifier) pair,
    the canonical person key is hashed. Otherwise falls back to a
    channel-scoped hash.
    """
    canonical = _resolve_canonical(channel, identifier)
    if canonical:
        return _hash_identity_key(canonical)
    return _hash_identity_key(f"{channel}:{identifier.strip()}")


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
        return _EMPTY_MAP
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return _EMPTY_MAP
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


def migrate_channel_hashes(seen_users: set[str]) -> int:
    """Add canonical person hashes to `seen_users` for any identity-map
    entry whose channel-scoped hash is already present. Mutates the set
    in place; returns the count added.

    Called on first state load after the identity map is rolled out, so
    that a deploy doesn't re-fire `user_onboarded` for every known user:
    before the map, seen_users held channel-scoped hashes; after the
    map, `get_distinct_id` returns canonical hashes, which would look
    "new" without this migration. Idempotent.
    """
    migrated = 0
    for channel_key, person_key in _load_identity_map().items():
        channel_hash = _hash_identity_key(channel_key)
        person_hash = _hash_identity_key(person_key)
        if channel_hash in seen_users and person_hash not in seen_users:
            seen_users.add(person_hash)
            migrated += 1
    return migrated
