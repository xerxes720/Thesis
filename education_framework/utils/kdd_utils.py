# kdd_utils.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import csv


def read_kdd_table(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Reads KDD Algebra .txt/.csv into a DataFrame.
    Tries C engine first (supports low_memory), falls back to python engine if necessary.
    """
    # 1) sniff delimiter (default tab)
    sniff_sep = sep
    if sniff_sep is None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(50_000)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=["\t", ",", ";", "|"])
                sniff_sep = dialect.delimiter
            except Exception:
                sniff_sep = "\t"
        except Exception:
            sniff_sep = "\t"

    # 2) try read with C engine first (fast + supports low_memory)
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                sep=sniff_sep,
                engine="c",
                low_memory=False,
                encoding=enc,
            )
        except Exception:
            pass

    # 3) fallback: python engine (no low_memory)
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                sep=sniff_sep,
                engine="python",
                encoding=enc,
            )
        except Exception:
            pass

    # 4) last resort
    return pd.read_csv(path, sep=sniff_sep)


def extract_primary_kc(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # KDD sometimes uses "~~" to separate multiple KCs
    if "~~" in s:
        s = s.split("~~")[0].strip()
    return s if s else None


def guess_order_cols(df: pd.DataFrame) -> List[str]:
    """
    Returns a best-effort list of columns to sort within each student.
    If none exist, caller should sort by original row order.
    """
    candidates = [
        "Row",
        "row",
        "Time",
        "time",
        "Step Start Time",
        "Step End Time",
        "Problem View",
        "Opportunity(Default)",
        "Opportunity",
    ]
    return [c for c in candidates if c in df.columns]


def group_rows_by_student_ordered(
        df: pd.DataFrame,
        *,
        student_col: str = "Anon Student Id",
        order_cols: Optional[Sequence[str]] = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Output: [(student_id, [row_dicts_sorted]), ...]
    Sorting: by order_cols if available, else by original CSV row order.
    """
    if student_col not in df.columns:
        raise ValueError(f"student_col not found: {student_col}")

    df2 = df.copy()

    # preserve stable input order
    df2["__row_idx__"] = range(len(df2))

    if order_cols:
        use_cols = [c for c in order_cols if c in df2.columns]
    else:
        use_cols = guess_order_cols(df2)

    sort_cols = [student_col] + (use_cols if use_cols else ["__row_idx__"])
    df2 = df2.sort_values(sort_cols, kind="mergesort")  # stable sort

    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    for sid, g in df2.groupby(student_col, sort=False):
        out.append((str(sid), g.drop(columns=["__row_idx__"]).to_dict(orient="records")))
    return out
