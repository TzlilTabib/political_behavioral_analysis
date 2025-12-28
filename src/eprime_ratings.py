"""
Docstring for behavioral_analyses.political_behavioral_analysis.src.eprime_ratings
Loads e-prime txt files including subject ratings of political stimuli
Creates a long-format dataframe with one row per subject per session per video type
"""

from pathlib import Path
import re
import pandas as pd

FILENAME_RE = re.compile(
    r"^.*-(?P<sub>\d{4})-(?P<video_type>[^-]+)-(?P<session>\d+)\.txt$"
)

VALUE_RE = re.compile(
    r"^\s*Stimulus\.Slider1\.Value:\s*(\d+)\s*$",
    re.MULTILINE
)

def parse_one_file(path: Path) -> dict:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Bad filename: {path.name}")

    # ✅ E-Prime text files are commonly UTF-16 (Windows)
    try:
        text = path.read_text(encoding="utf-16")
    except UnicodeError:
        # fallback if you ever have a weird file
        text = path.read_text(encoding="utf-8", errors="ignore")

    values = [int(v) for v in VALUE_RE.findall(text)]

    if len(values) < 5:
        raise ValueError(f"{path.name}: expected 5 values, got {len(values)}")
    if len(values) > 5:
        values = values[:5]

    return {
        "subject": m.group("sub"),
        "video_type": m.group("video_type"),
        "session": int(m.group("session")),
        "q1_share": values[0],
        "q2_support": values[1],
        "q3_emotional": values[2],
        "q4_interesting": values[3],
        "q5_extreme": values[4],
        "source_file": path.name,
    }


def build_long_df(
    input_dir: str | Path,
    questionnaire_prefix: str = "Potilical_views_questionnaire",
) -> pd.DataFrame:
    input_dir = Path(input_dir)

    # Only questionnaire files, ignore Emotional_Localizer etc.
    paths = sorted(input_dir.rglob(f"{questionnaire_prefix}-*.txt"))

    rows = []
    errors = []
    for p in paths:
        try:
            rows.append(parse_one_file(p))
        except Exception as e:
            errors.append((p.name, str(e)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["subject", "session", "video_type"]).reset_index(drop=True)

    if errors:
        print("⚠️ Some questionnaire files could not be parsed:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")

    return df
