"""
Political attitude questionnaire utilities.

Designed for Qualtrics-style TSV exports (often UTF-16 encoded) and the
YY study subject codes (either "00XX" or "YY_PL_xx").

Main entry point
----------------
load_political_attitudes_tsv(path) -> pd.DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import pandas as pd


# ---------------------------
# Subject code normalization
# ---------------------------

_SUBJECT_CLEAN_RE = re.compile(r"[^A-Za-z0-9_]+")


def normalize_subject_code(raw: object) -> Optional[str]:
    """
    Normalize a subject code string.

    Examples
    --------
    " YY_PL_15 " -> "YY_PL_15"
    "\"YY_PL_15\"" -> "YY_PL_15"
    "0003" -> "0003"
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()

    # strip wrapping quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # remove odd symbols (keep letters, digits, underscore)
    s = _SUBJECT_CLEAN_RE.sub("", s)

    # collapse repeated underscores
    s = re.sub(r"_+", "_", s).strip("_")

    return s or None


def extract_subject_num(code: Optional[str]) -> Optional[int]:
    """
    Extract the numeric ID from a normalized subject code.

    Rules
    -----
    - "YY_PL_02" -> 2
    - "0003"     -> 3
    - "YYPL03"   -> 3  (after normalization symbols are removed, underscores preserved when present)

    Returns None if no digits found.
    """
    if not code:
        return None

    m = re.search(r"(\d+)$", code)
    if not m:
        return None
    return int(m.group(1))


def make_yy_pl_code(subject_num: Optional[int]) -> Optional[str]:
    """
    Make canonical YY_PL_XX code from numeric ID.
    - 3 -> YY_PL_03
    - 12 -> YY_PL_12
    - 105 -> YY_PL_105  (keeps full number; padding doesn't truncate)
    """
    if subject_num is None or (isinstance(subject_num, float) and pd.isna(subject_num)):
        return None
    return f"YY_PL_{int(subject_num):02d}"


# ---------------------------
# Qualtrics export cleaning
# ---------------------------

def _read_tsv_best_effort(path: str | Path) -> pd.DataFrame:
    """
    Read TSV with robust encoding handling.
    Qualtrics TSVs are often UTF-16; fallback to UTF-8.
    """
    path = Path(path)
    try:
        return pd.read_csv(path, sep="\t", encoding="utf-16")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep="\t", encoding="utf-8")


def _drop_qualtrics_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop Qualtrics 'question text' row and 'ImportId' row if present.
    Heuristic: first two rows often contain non-data strings in 'Progress'.
    """
    out = df.copy()

    # Coerce Progress to numeric where possible
    prog = pd.to_numeric(out.get("Progress"), errors="coerce")

    # Keep rows that have a numeric Progress (e.g., 0-100)
    out = out.loc[prog.notna()].copy()

    # Recompute Progress as int where possible
    out["Progress"] = pd.to_numeric(out["Progress"], errors="coerce").astype("Int64")

    return out


# ---------------------------
# Public API
# ---------------------------

@dataclass(frozen=True)
class PoliticalAttitudesColumns:
    subject_raw: str = "Q3"
    progress: str = "Progress"
    political_engagement: str = "Q7_1"
    camp_support: str = "Q9_1"
    government_support: str = "Q10_1"
    coalition_opposition_support: str = "Q11_1"
    vote_choice: str = "Q5"
    vote_choice_text: str = "Q5_12_TEXT"


def load_political_attitudes_tsv(
    path: str | Path,
    *,
    cols: PoliticalAttitudesColumns = PoliticalAttitudesColumns(),
    keep_only_progress_100: bool = True,
    save_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load and clean the political attitude questionnaire TSV.

    Cleaning steps
    --------------
    1) Read TSV (tries UTF-16, then UTF-8)
    2) Drop Qualtrics header rows (question text / ImportId)
    3) Optionally filter Progress == 100
    4) Normalize subject codes (remove quotes/symbols, trim spaces)
    5) Two subject columns:
        - subject_code_yy : canonical "YY_PL_XX" code (also for numeric-only inputs like "0003")
        - subject_num     : int extracted from the code
    6) Create and return a slim analysis-ready dataframe including:
        subject_* + engagement/support scores + vote choice text

    Returns
    -------
    pd.DataFrame
        Columns:
          subject_code_yy, subject_num,
          political_engagement, camp_support, government_support,
          coalition_opposition_support, vote_choice, vote_choice_text,
          progress
    """
    df = _read_tsv_best_effort(path)
    df = _drop_qualtrics_header_rows(df)

    if keep_only_progress_100 and cols.progress in df.columns:
        df = df.loc[df[cols.progress] == 100].copy()

    # Normalize subject code (raw)
    df["_subject_code_norm"] = df[cols.subject_raw].map(normalize_subject_code)

    # ✅ CHANGED: derive numeric ID then canonical YY code
    df["subject_num"] = df["_subject_code_norm"].map(extract_subject_num).astype("Int64")
    df["subject_code_yy"] = df["subject_num"].map(make_yy_pl_code)

    # Build analysis-ready view (keep original Progress for QA)
    out_cols = {
        "subject_code_yy": "subject_code_yy",
        "subject_num": "subject_num",
        cols.political_engagement: "political_engagement",
        cols.camp_support: "camp_support",
        cols.government_support: "government_support",
        cols.coalition_opposition_support: "coalition_opposition_support",
        cols.vote_choice: "vote_choice",
        cols.vote_choice_text: "vote_choice_text",
        cols.progress: "progress",
    }

    # Only keep columns that exist (safer if Qualtrics export changes)
    present = {k: v for k, v in out_cols.items() if k in df.columns}
    out = df[list(present.keys())].rename(columns=present)

    # Save if requested
    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_csv, index=False)

    return out

# ---------------------------
# QC / Debugging utility
# ---------------------------

def get_removed_rows_report(
    path: str | Path,
    *,
    cols: PoliticalAttitudesColumns = PoliticalAttitudesColumns(),
    keep_only_progress_100: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    🆕 DEBUGGING UTILITY

    Return a report of rows removed during cleaning.

    Returns
    -------
    dict with keys:
      - "removed_non_numeric_progress"
      - "removed_progress_not_100" (only if keep_only_progress_100=True)
    """

    df_raw = _read_tsv_best_effort(path)

    report: dict[str, pd.DataFrame] = {}

    # ----------------------------
    # Step 1: Qualtrics header rows
    # ----------------------------
    prog_numeric = pd.to_numeric(df_raw.get(cols.progress), errors="coerce")

    removed_non_numeric = df_raw.loc[prog_numeric.isna()].copy()
    report["removed_non_numeric_progress"] = removed_non_numeric

    df_clean = df_raw.loc[prog_numeric.notna()].copy()
    df_clean[cols.progress] = prog_numeric.loc[prog_numeric.notna()].astype("Int64")

    # ----------------------------
    # Step 2: Progress != 100
    # ----------------------------
    if keep_only_progress_100 and cols.progress in df_clean.columns:
        removed_not_100 = df_clean.loc[df_clean[cols.progress] != 100].copy()
        report["removed_progress_not_100"] = removed_not_100

    return report
