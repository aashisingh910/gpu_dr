"""dr.csv_export - best-effort CSV companions for this project's JSON reports.

Every script under scripts/ already writes a JSON file as the authoritative
record of what it computed - nothing here changes that. This module just
gives the same numbers a form that opens directly in Excel/a spreadsheet or
skims in a plain text editor without parsing nested JSON by eye, wherever the
underlying data has one obvious tabular shape:

    list of flat dicts          -> one row per entry            (e.g. per-epoch
                                                                   training history)
    dict of dicts                -> one row per top-level key    (e.g. per-group
                                                                   evaluation metrics,
                                                                   per-model comparison)
    flat dict of scalars          -> a single-row CSV             (e.g. one report's
                                                                   summary numbers)

Anything more irregular (deep nesting mixed with per-sample arrays, etc.) is
left as JSON-only rather than guessing a misleading flat layout - the
functions below say so on stdout when that happens, they do not fail the
calling script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def save_csv_alongside(obj: Any, json_path: str | Path, csv_path: str | Path | None = None) -> bool:
    """Write `obj` (whatever a report script is also saving as JSON) to a CSV
    at `csv_path` (default: `json_path` with a .csv suffix), if its shape is
    tabular enough for that to make sense.

    Returns True if a CSV was written, False if the shape was skipped or the
    optional pandas dependency is unavailable (this project already depends
    on pandas elsewhere, so the latter should not happen in practice).
    """
    try:
        import pandas as pd
    except ImportError:                                       # noqa: BLE001
        return False

    csv_path = Path(csv_path) if csv_path is not None else Path(json_path).with_suffix(".csv")

    try:
        if isinstance(obj, list):
            if not obj or not all(isinstance(x, dict) for x in obj):
                return False
            df = pd.json_normalize(obj)
        elif isinstance(obj, dict):
            if not obj:
                return False
            if all(isinstance(v, dict) for v in obj.values()):
                # a dict of dicts (per-group/per-model/per-class results) ->
                # one named row per top-level key
                df = pd.json_normalize([{"name": k, **v} for k, v in obj.items()])
            elif any(isinstance(v, dict) for v in obj.values()):
                # a mix of scalar and nested fields at the top level - flatten
                # the whole thing into one row rather than skip it
                df = pd.json_normalize([obj])
            else:
                df = pd.json_normalize([obj])
        else:
            return False
        if df.empty:
            return False
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:                                    # noqa: BLE001
        print(f"[csv] could not flatten {json_path} to a CSV ({type(e).__name__}: {e}); "
              f"the JSON file remains the full record")
        return False
