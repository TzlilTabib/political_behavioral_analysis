from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable
import pandas as pd
from pathlib import Path
import unicodedata

# Load data
def load_qualtrics_tsv(path: str | Path, subject_col: str, subject_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-16")
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = [
        unicodedata.normalize("NFKC", c)
        .replace("\xa0", " ")
        .strip()
        for c in df.columns
    ]
    assert not any("\xa0" in c for c in df.columns), "NBSP found in column names"

    # ---- subject filtering
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found in TSV")

    df = df[df[subject_col].astype(str).str.contains(subject_prefix, na=False)]

    df = df.reset_index(drop=True)

    return df

# Specs
@dataclass(frozen=True)
class ScoreSpec:
    name: str
    construct: str
    description: str
    scale: str
    inputs: list[str]
    formula: str
    notes: str = ""

# Optional: keep a registry for codebook export
SPECS: list[ScoreSpec] = [
    ScoreSpec(
        name="affpol_thermo",
        construct="Affective polarization (thermometer)",
        description="In-party warmth minus out-party warmth (0–100).",
        scale="~[-100, 100]",
        inputs=["feelings to right_1", "Q43_1", "Political_affiliatio", "Center/Other"],
        formula="thermo_inparty - thermo_outparty",
        notes="Center participants treated as leaners using Center/Other.",
    ),
    ScoreSpec(
        name="outparty_social_distance",
        construct="Social distance / avoidance",
        description="Mean of (reversed out-party friend comfort, reversed out-party neighbor comfort, out-party intermarriage upset).",
        scale="[1, 4] approx",
        inputs=["Right_Comfort", "Left_Comfort", "right_neighbors", "left_neighbors", "right_family", "left_family"],
        formula="mean([5-friend_comfort_out, 5-neigh_comfort_out, marriage_upset_out])",
        notes="Comfort items reversed so higher = more rejection; upset is already higher=more.",
    ),
        ScoreSpec(
        name="party_identity_strength",
        construct="Partisan identity strength",
        description="Mean of in-party identity importance, being in-party, in-party describes me, we-vs-they, and extremity strength.",
        #  FIX: your current computation mixes 1–5, 1–4, and 1–2 scales
        scale="mixed (approx 1–4.2)",  
        inputs=[
            "Left_identity", "right__identity", "center/other_identit",
            "left_being", "right_being", "center/other_being",
            "left_describe", "right_describe", "center/othe_describe",
            "we_or_they",
            "Left_Extrem", "Right_Extreme"
        ],
        formula="mean([identity_in, being_in, describe_in, we_they(1–4), extremity_strength(1–2)])",  
        notes=" Not rescaled. If you want a true 1–5 scale, rescale we_they and extremity before averaging.",
    ),
    ScoreSpec(
        name="trait_bias",
        construct="Trait attribution bias",
        description="Combines positive-trait bias (pos_in - pos_out) and negative-trait bias (neg_out - neg_in).",
        scale="~[-4, 4]",
        inputs=["right_table_1..8", "left_table_1..8", "Political_affiliatio", "Center/Other"],
        formula="mean([pos_in_minus_out, neg_out_minus_in])",
        notes="Trait order per questionnaire table: patriotic,intelligent,honest,open-minded,generous,hypocritical,selfish,mean.",
    ),
    ScoreSpec(
        name="trust_bias",
        construct="Trust bias",
        description="Trust in in-party minus trust in out-party.",
        scale="[-4, 4]",
        inputs=["right_trust", "left_trust", "Political_affiliatio", "Center/Other"],
        formula="trust_inparty - trust_outparty",
    ),
    # Unfavorable thoughts spec 
    ScoreSpec(
        name="thought_intensity_bias_unfav",
        construct="Thought intensity (unfavorable)",
        description="Unfavorable intensity out-party minus in-party (Hebrew 4-level intensity).",  
        scale="[-3, 3]",  
        inputs=[
            "right_one_negative", "left_one_negative",
            "Political_affiliatio", "Center/Other",
        ],  
        formula="unfav_outparty - unfav_inparty",
        notes=" Direct intensity labels in this export (no separate presence+followup columns used).",
    ),
    # Favorable thoughts bias spec
    ScoreSpec(
        name="thought_intensity_bias_fav",
        construct="Thought intensity (favorable)",
        description="Favorable intensity in-party minus out-party (Hebrew 4-level intensity).",
        scale="[-3, 3]",
        inputs=[
            "right_one_positive", "left_one_positive",
            "Political_affiliatio", "Center/Other",
        ],
        formula="fav_inparty - fav_outparty",
        notes=" Direct intensity labels in this export.",
    ),
]

# Code book

def build_codebook_markdown() -> str:
    lines = []
    lines.append("# Polarization Questionnaire Scoring Codebook\n")
    lines.append("| Score name | Construct | Description | Scale | Inputs | Formula | Notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in SPECS:
        inputs = ", ".join(s.inputs)
        lines.append(f"| `{s.name}` | {s.construct} | {s.description} | {s.scale} | {inputs} | {s.formula} | {s.notes} |")
    lines.append("")
    return "\n".join(lines)

def write_codebook(path: str | Path) -> Path:
    path = Path(path)
    path.write_text(build_codebook_markdown(), encoding="utf-8")
    return path

# Party
def infer_camp(row, center_policy: str = "lean") -> str | None:
    """
    Returns respondent's effective camp:
      - "שמאל" or "ימין" for left/right
      - for "מרכז" (center):
          - "lean": use Center/Other to map to left/right when possible
          - "exclude": returns None
          - "center": returns "מרכז"
    """
    base = row.get("Political_affiliatio")

    if base in ("שמאל", "ימין"):
        return base

    if base == "מרכז":
        if center_policy == "exclude":
            return None
        if center_policy == "center":
            return "מרכז"

        # default: lean
        lean = row.get("Center/Other")
        if lean == "קרוב/ה יותר למחנה השמאל":
            return "שמאל"
        if lean == "קרוב/ה יותר למחנה הימין":
            return "ימין"
        return "מרכז"

    return None


def in_out_labels(camp: str | None) -> tuple[str | None, str | None]:
    """
    Converts camp to in/out labels used throughout scoring.
    Returns ("left","right") or ("right","left"), else (None,None).
    """
    if camp == "שמאל":
        return ("left", "right")
    if camp == "ימין":
        return ("right", "left")
    return (None, None)


# Mapping

# Generic 5-point Likert (adjust keys if your exact Hebrew strings differ)
LIKERT_5_HE = {
    "כלל לא": 1,
    "בכלל לא": 1,
    "מעט מאוד": 2,
    "במידה מסוימת": 3,
    "במידה רבה": 4,
    "במידה רבה מאוד": 5,
}

# Trust frequency (Wave 2 trust items)
TRUST_5_HE = {
    "כמעט אף פעם": 1,
    "מידי פעם": 2,
    "מחצית מהזמן": 3,
    "רוב הזמן": 4,
    "כמעט תמיד": 5,
}

# Comfort (Wave 1 friend/neighbors)
COMFORT_4_HE = {
    "כלל לא בנוח": 1,
    "כלל לא נוח": 1,
    "לא כל-כך בנוח": 2,
    "לא כל-כך נוח": 2,
    "די בנוח": 3,
    "מאוד בנוח": 4,
}

# Upset about intermarriage (Wave 1)
UPSET_4_HE = {
    "כלל לא מוטרד/ת": 1,
    "לא כל-כך מוטרד/ת": 2,
    "די מוטרד/ת": 3,
    "מאוד מוטרד/ת": 4,
}

# “We” vs “They” frequency (Wave 1)
WE_THEY_4_HE = {
    "אף פעם": 1,
    "לעיתים רחוקות": 2,
    "לפעמים": 3,
    "רוב הזמן": 4,
}



# Scoring

TRAITS = [
    "patriotic", "intelligent", "honest", "open_minded", "generous",
    "hypocritical", "selfish", "mean",
]
POS_TRAITS = {"patriotic","intelligent","honest","open_minded","generous"}
NEG_TRAITS = {"hypocritical","selfish","mean"}

def _to_num(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping)

def _mean_row(values):
    values = [v for v in values if pd.notna(v)]
    return sum(values)/len(values) if values else pd.NA

def compute_scores(df: pd.DataFrame, center_policy="lean") -> pd.DataFrame:
    out = df.copy()

    # ---- camp + in/out labels
    out["camp"] = out.apply(lambda r: infer_camp(r, center_policy=center_policy), axis=1)
    # camp is "שמאל"/"ימין"/"מרכז"/None
    # we compute in/out only for left/right
    out["in_label"] = out["camp"].map({"שמאל": "left", "ימין": "right"})
    out["out_label"] = out["camp"].map({"שמאל": "right", "ימין": "left"})

    # ---- thermometer
    out["thermo_right"] = pd.to_numeric(out.get("feelings to right_1"), errors="coerce")
    out["thermo_left"]  = pd.to_numeric(out.get("Q43_1"), errors="coerce")

    out["thermo_inparty"]  = out.apply(lambda r: r["thermo_left"] if r["camp"]=="שמאל" else (r["thermo_right"] if r["camp"]=="ימין" else pd.NA), axis=1)
    out["thermo_outparty"] = out.apply(lambda r: r["thermo_right"] if r["camp"]=="שמאל" else (r["thermo_left"]  if r["camp"]=="ימין" else pd.NA), axis=1)
    out["affpol_thermo"] = out["thermo_inparty"] - out["thermo_outparty"]

    # ---- social distance
    out["comfort_left_friend"]  = _to_num(out.get("Right_Comfort"), COMFORT_4_HE)   # friend is left
    out["comfort_right_friend"] = _to_num(out.get("Left_Comfort"), COMFORT_4_HE)    # friend is right
    out["comfort_left_neigh"]   = _to_num(out.get("right_neighbors"), COMFORT_4_HE) # neighbor is left
    out["comfort_right_neigh"]  = _to_num(out.get("left_neighbors"), COMFORT_4_HE)  # neighbor is right
    out["upset_left_marry"]     = _to_num(out.get("right_family"), UPSET_4_HE)      # marry left
    # NOTE: some exports contain NBSP; handle both
    lf = "left_family"
    lf_nbsp = "left_family\u00a0"
    out["upset_right_marry"]    = _to_num(out.get(lf) if lf in out.columns else out.get(lf_nbsp), UPSET_4_HE)

    def _outparty_sd(r):
        if r["camp"] == "שמאל":
            friend_rej = 5 - r["comfort_right_friend"] if pd.notna(r["comfort_right_friend"]) else pd.NA
            neigh_rej  = 5 - r["comfort_right_neigh"]  if pd.notna(r["comfort_right_neigh"])  else pd.NA
            marry_rej  = r["upset_right_marry"]
        elif r["camp"] == "ימין":
            friend_rej = 5 - r["comfort_left_friend"] if pd.notna(r["comfort_left_friend"]) else pd.NA
            neigh_rej  = 5 - r["comfort_left_neigh"]  if pd.notna(r["comfort_left_neigh"])  else pd.NA
            marry_rej  = r["upset_left_marry"]
        else:
            return pd.NA
        return _mean_row([friend_rej, neigh_rej, marry_rej])

    out["outparty_social_distance"] = out.apply(_outparty_sd, axis=1)

    # ---- trust bias
    out["trust_right"] = _to_num(out.get("right_trust"), TRUST_5_HE)
    out["trust_left"]  = _to_num(out.get("left_trust"), TRUST_5_HE)
    out["trust_inparty"]  = out.apply(lambda r: r["trust_left"] if r["camp"]=="שמאל" else (r["trust_right"] if r["camp"]=="ימין" else pd.NA), axis=1)
    out["trust_outparty"] = out.apply(lambda r: r["trust_right"] if r["camp"]=="שמאל" else (r["trust_left"]  if r["camp"]=="ימין" else pd.NA), axis=1)
    out["trust_bias"] = out["trust_inparty"] - out["trust_outparty"]

    # ---- party identity strength (in-party piped items)
    # Map relevant columns to numeric
    id_left   = _to_num(out.get("Left_identity"), LIKERT_5_HE)
    id_right  = _to_num(out.get("right__identity"), LIKERT_5_HE)
    id_center = _to_num(out.get("center/other_identit"), LIKERT_5_HE)

    being_left   = _to_num(out.get("left_being"), LIKERT_5_HE)
    being_right  = _to_num(out.get("right_being"), LIKERT_5_HE)
    being_center = _to_num(out.get("center/other_being"), LIKERT_5_HE)

    desc_left   = _to_num(out.get("left_describe"), LIKERT_5_HE)
    desc_right  = _to_num(out.get("right_describe"), LIKERT_5_HE)
    desc_center = _to_num(out.get("center/othe_describe"), LIKERT_5_HE)

    out["we_they"] = _to_num(out.get("we_or_they"), WE_THEY_4_HE)

    # Extremity strength: depends on which side (may be 2-point "strong/not strong")
    # If your export encodes it differently, you can refine this mapping.
    EXTREM_2 = {"Strong": 2, "Not very strong": 1, "חזק/ה": 2, "לא כל-כך חזק/ה": 1}
    extrem_left  = out.get("Left_Extrem").map(EXTREM_2) if "Left_Extrem" in out.columns else pd.Series(pd.NA, index=out.index)
    extrem_right = out.get("Right_Extreme").map(EXTREM_2) if "Right_Extreme" in out.columns else pd.Series(pd.NA, index=out.index)

    def _pick_inparty(r, left_val, right_val, center_val):
        if r["camp"] == "שמאל": return left_val
        if r["camp"] == "ימין": return right_val
        # center treated as leaners already; if still "מרכז", return center val
        if r["camp"] == "מרכז": return center_val
        return pd.NA

    out["identity_in"] = out.apply(lambda r: _pick_inparty(r, id_left.loc[r.name], id_right.loc[r.name], id_center.loc[r.name]), axis=1)
    out["being_in"]    = out.apply(lambda r: _pick_inparty(r, being_left.loc[r.name], being_right.loc[r.name], being_center.loc[r.name]), axis=1)
    out["describe_in"] = out.apply(lambda r: _pick_inparty(r, desc_left.loc[r.name], desc_right.loc[r.name], desc_center.loc[r.name]), axis=1)
    out["extremity_strength"] = out.apply(lambda r: extrem_left.loc[r.name] if r["camp"]=="שמאל" else (extrem_right.loc[r.name] if r["camp"]=="ימין" else pd.NA), axis=1)

    out["party_identity_strength"] = out.apply(
        lambda r: _mean_row([r["identity_in"], r["being_in"], r["describe_in"], r["we_they"], r["extremity_strength"]]),
        axis=1
    )

    # ---- traits: right_table_1..8 and left_table_1..8
    # Assume each cell is already 1-5. If Hebrew labels exist, map with LIKERT_5_HE.
    def _get_trait_block(prefix: str):
        cols = [f"{prefix}_table_{i}" for i in range(1, 9)]
        block = out[cols].copy()
        # If strings, attempt LIKERT_5_HE; otherwise coerce numeric
        for c in cols:
            if block[c].dtype == object:
                block[c] = block[c].map(LIKERT_5_HE)
            block[c] = pd.to_numeric(block[c], errors="coerce")
        block.columns = TRAITS
        return block

    right_traits = _get_trait_block("right")
    left_traits  = _get_trait_block("left")

    out["pos_right"] = right_traits[[t for t in TRAITS if t in POS_TRAITS]].mean(axis=1)
    out["neg_right"] = right_traits[[t for t in TRAITS if t in NEG_TRAITS]].mean(axis=1)
    out["pos_left"]  = left_traits[[t for t in TRAITS if t in POS_TRAITS]].mean(axis=1)
    out["neg_left"]  = left_traits[[t for t in TRAITS if t in NEG_TRAITS]].mean(axis=1)

    out["pos_inparty"]  = out.apply(lambda r: r["pos_left"]  if r["camp"]=="שמאל" else (r["pos_right"] if r["camp"]=="ימין" else pd.NA), axis=1)
    out["pos_outparty"] = out.apply(lambda r: r["pos_right"] if r["camp"]=="שמאל" else (r["pos_left"]  if r["camp"]=="ימין" else pd.NA), axis=1)
    out["neg_inparty"]  = out.apply(lambda r: r["neg_left"]  if r["camp"]=="שמאל" else (r["neg_right"] if r["camp"]=="ימין" else pd.NA), axis=1)
    out["neg_outparty"] = out.apply(lambda r: r["neg_right"] if r["camp"]=="שמאל" else (r["neg_left"]  if r["camp"]=="ימין" else pd.NA), axis=1)

    out["pos_in_minus_out"] = out["pos_inparty"] - out["pos_outparty"]
    out["neg_out_minus_in"] = out["neg_outparty"] - out["neg_inparty"]
    out["trait_bias"] = (out["pos_in_minus_out"] + out["neg_out_minus_in"]) / 2

    # ---- thought intensity (presence + intensity -> 0..4)
    # You may need to adjust the Hebrew strings here after you inspect unique values.
    UNFAV_INTENS_4_HE = {
        "שליליות במידה מועטה": 1,
        "שליליות במידה בינונית": 2,
        "שליליות מאוד": 3,
        "שליליות במידה רבה מאוד": 4,
    }
    FAV_INTENS_4_HE = {
        "חיוביות במידה מועטה": 1,
        "חיוביות במידה בינונית": 2,
        "חיוביות מאוד": 3,
        "חיוביות במידה רבה מאוד": 4,
    }


    out["unfav_right"] = out["right_one_negative"].map(UNFAV_INTENS_4_HE)
    out["unfav_left"] = out["left_one_negative"].map(UNFAV_INTENS_4_HE)

    out["unfav_inparty"] = out.apply(
        lambda r: r["unfav_left"] if r["camp"] == "שמאל"
        else (r["unfav_right"] if r["camp"] == "ימין" else pd.NA),
        axis=1
    )
    out["unfav_outparty"] = out.apply(
        lambda r: r["unfav_right"] if r["camp"] == "שמאל"
        else (r["unfav_left"] if r["camp"] == "ימין" else pd.NA),
        axis=1
    )
    out["thought_intensity_bias_unfav"] = out["unfav_outparty"] - out["unfav_inparty"]

    out["fav_right"] = out["right_one_positive"].map(FAV_INTENS_4_HE)
    out["fav_left"] = out["left_one_positive"].map(FAV_INTENS_4_HE)

    out["fav_inparty"] = out.apply(
        lambda r: r["fav_left"] if r["camp"] == "שמאל"
        else (r["fav_right"] if r["camp"] == "ימין" else pd.NA),
        axis=1
    )
    out["fav_outparty"] = out.apply(
        lambda r: r["fav_right"] if r["camp"] == "שמאל"
        else (r["fav_left"] if r["camp"] == "ימין" else pd.NA),
        axis=1
    )
    out["thought_intensity_bias_fav"] = out["fav_inparty"] - out["fav_outparty"]

    return out

