"""Trajectories held out from evaluation because the method was developed on them.

`experiment.py` filters these out of any input export, so a cohort assembled from a fresh
download is automatically disjoint from the development set without the reproducer having
to know which trajectories those were. The identifiers live in development_cohort.json.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "development_cohort.json"


@lru_cache(maxsize=1)
def development_ids() -> dict[str, frozenset[str]]:
    """Development-cohort identifiers per split. Splits with no entries are absent."""

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        split: frozenset(str(value) for value in entry["ids"])
        for split, entry in payload.items()
        if isinstance(entry, dict) and entry.get("ids")
    }


def is_development(split_name: str, trajectory_id: str) -> bool:
    return trajectory_id in development_ids().get(split_name, frozenset())
