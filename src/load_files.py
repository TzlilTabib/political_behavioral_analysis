"""
Functions to load tabular data files.

(1) load_metadata: Loads data from a local CSV or Excel file, including subject IDs and related metadata (scan date and time, age at scan, gender).
(2) load_political_attitude: Loads political attitude questionnaire data from a local CSV or Excel file (wide format).
(3) load_political_polarization: Loads political polarization questionnaire data from a local CSV or Excel file (wide format)
(4) load_posts_ratings: Loads social media posts ratings data from a local CSV or Excel file (4 files per subject).

Uses resolve_ids: Utilizes the resolve_ids module to map between different subject identifiers.

"""

import pandas as pd
from src.resolve_ids import resolve_subject_ids
import os

def load_metadata(file_path):
    """
    Loads metadata from a CSV or Excel file.

    Args:
        file_path (str): Path to the metadata file.
    Returns:
        pd.DataFrame: DataFrame containing the metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    if file_path.endswith('.csv'):
        metadata = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        metadata = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    return metadata    