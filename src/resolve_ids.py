"""
Normalizes and resolves IDs for subjects - 
subjects can be identified by multiple different IDs - 
(1) 00XX
(2) YY_PL_X

Subjects are also recognized by their specific scan date and time, 
and BIDS assigns a unique identifier to each ("sub-X")
This module helps to map between these different identifiers.

"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# ==========================================================================
# OPTIONAL: MERGING BEHAVIORAL DATA WITH BIDS FILES IDS BASED ON SESSION ID
# --------------------------------------------------------------------------
# by scanning the BIDS tree
# ==========================================================================

def build_bids_mapping(
    bids_root: str | Path,
) -> pd.DataFrame:
    """
    Scan a BIDS tree and return a DataFrame with:
    - bids_id (e.g. 'sub-01')
    - ses_id  (e.g. '20240215T0930', taken from 'ses-20240215T0930')
    
    Assumes exactly one session per subject.
    """
    bids_root = Path(bids_root)
    rows = []

    for sub_dir in sorted(bids_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        bids_id = sub_dir.name  # 'sub-01'

        ses_dirs = sorted(sub_dir.glob("ses-*"))
        if len(ses_dirs) == 0:
            print(f"⚠️ No ses-* folder for {bids_id}, skipping.")
            continue
        if len(ses_dirs) > 1:
            print(f"⚠️ More than one ses-* folder for {bids_id}: {ses_dirs}")
            # If this ever happens, decide how to handle it.
            ses_dir = ses_dirs[0]
        else:
            ses_dir = ses_dirs[0]

        ses_label = ses_dir.name.replace("ses-", "")  # remove 'ses-' prefix
        rows.append({"bids_id": bids_id, "ses_id": ses_label})

    return pd.DataFrame(rows)


def attach_bids_ids_to_behav(
    bids_root: str | Path,
    behav_csv: str | Path,
    output_csv: str | Path,
    csv_session_col: str = "ses_id",
) -> pd.DataFrame:
    """
    Merge behavioral CSV with BIDS info based on session ID and save a new CSV.

    Parameters
    ----------
    bids_root : path to BIDS root (with sub-XX/ses-YY folders)
    behav_csv : path to CSV with at least a 'ses_id' column (or rename via csv_session_col)
    output_csv : where to save the enriched CSV
    csv_session_col : the column name in behav_csv holding the session ID
                      that corresponds to the part after 'ses-' in BIDS.
    """
    # 1) BIDS mapping: bids_id + ses_id
    df_bids = build_bids_mapping(bids_root)
    df_bids["ses_id"] = df_bids["ses_id"].astype(str)

    # 2) Load your behavioral info
    df_behav = pd.read_csv(behav_csv)
    df_behav[csv_session_col] = df_behav[csv_session_col].astype(str)
    
    # 3) Merge on session id
    df_merged = df_behav.merge(
        df_bids,
        left_on=csv_session_col,
        right_on="ses_id",
        how="left",
        validate="one_to_one",
    )

    # Sanity check: who didn't get a bids_id?
    missing = df_merged["bids_id"].isna()
    if missing.any():
        print("⚠️ Some behavioral rows have no matching BIDS session:")
        print(df_merged.loc[missing, [csv_session_col]].drop_duplicates())

    # 4) Save
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    print(f"✅ Saved merged CSV with BIDS IDs to: {output_csv}")

    return df_merged

# ====================================================================
# OPTIONAL: MERGING BEHAVIORAL DATA WITH MANIFEST BASED ON SESSION ID
# --------------------------------------------------------------------
# using a DICOM manifest file
# ====================================================================

def make_bids_id(subject_code: str) -> str:
    """
    Create a BIDS subject id from the manifest subject_code.

    Examples:
    ---------
    '001'      -> 'sub-001'
    'YY_PL_1'  -> 'sub-YY_PL_1'

    We DO NOT strip zeros or change the code – we just prefix with 'sub-'.
    """
    if pd.isna(subject_code):
        return pd.NA
    return "sub-" + str(subject_code)

def merge_behavior_with_manifest_by_session(
    manifest_tsv: str | Path,
    behavioral_csv: str | Path,
    output_csv: str | Path,
    manifest_subject_col: str = "subject_code",
    manifest_session_col: str = "session_id",
    behav_session_col: str = "session_id",
) -> pd.DataFrame:
    """
    Merge behavioral data with DICOM manifest based on session_id and
    add a 'bids_id' column derived from the manifest subject_code.

    Parameters
    ----------
    manifest_tsv : path to DICOM manifest (TSV)
        Must contain at least [manifest_subject_col, manifest_session_col].
    behavioral_csv : path to behavioral CSV
        Must contain behav_session_col (same logical session id).
    output_csv : where to save the merged behavioral file.
    manifest_subject_col : column in manifest with the subject code used for BIDS.
    manifest_session_col : column in manifest with session id.
    behav_session_col : column in behavioral CSV with session id.

    Returns
    -------
    df_merged : DataFrame
        Behavioral rows + columns from manifest + 'bids_id'.
    """
    manifest_tsv = Path(manifest_tsv)
    behavioral_csv = Path(behavioral_csv)
    output_csv = Path(output_csv)

    # ---- 1) Load files ----
    df_manifest = pd.read_csv(manifest_tsv, sep="\t")
    df_behav = pd.read_csv(behavioral_csv, sep=",")

    # ---- 2) Sanity checks on columns ----
    for col, df, name in [
        (manifest_subject_col, df_manifest, "manifest"),
        (manifest_session_col, df_manifest, "manifest"),
        (behav_session_col, df_behav, "behavioral"),
    ]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {name} file. "
                f"Available columns: {list(df.columns)}"
            )
   # ---- 3) Create bids_id in the MANIFEST ----
    df_manifest[manifest_session_col] = df_manifest[manifest_session_col].astype(str)
    df_manifest["bids_id"] = df_manifest[manifest_subject_col].apply(make_bids_id)

    # We only keep session_id + bids_id for the merge
    df_manifest_small = df_manifest[[manifest_session_col, "bids_id"]].drop_duplicates()

    # ---- 4) Prepare behavioral table ----
    df_behav[behav_session_col] = df_behav[behav_session_col].astype(str)

    # ---- 5) Merge on session id ONLY ----
    df_merged = df_behav.merge(
        df_manifest_small,
        left_on=behav_session_col,
        right_on=manifest_session_col,
        how="left",
        validate="one_to_one",
    )

    # ---- 6) Save and return ----
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    print(f"✅ Saved merged behavioral file with BIDS IDs → {output_csv}")

    return df_merged
