"""
Functions to load tabular data files.

(1) load_metadata: Loads data from a local CSV or Excel file, including subject IDs and related metadata (scan date and time, age at scan, gender, political camps at recruitment).
(2) load_political_attitude: Loads political attitude questionnaire data from a local CSV or Excel file (wide format).
(3) load_political_polarization: Loads political polarization questionnaire data from a local CSV or Excel file (wide format)
(4) load_posts_ratings: Loads social media posts ratings data from a local CSV or Excel file (4 files per subject, post type and subject code - 00XX are in file name).

Uses resolve_ids: Utilizes the resolve_ids module to map between different subject identifiers.

"""

import pandas as pd
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

def load_political_attitude(file_path):
    """
    Loads political attitude questionnaire data from a CSV or Excel file.

    Args:
        file_path (str): Path to the political attitude data file.
    Returns:
        pd.DataFrame: DataFrame containing the political attitude data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Political attitude file not found: {file_path}")

    if file_path.endswith('.csv'):
        attitude_data = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        attitude_data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    return attitude_data

def load_political_polarization(file_path):
    """
    Loads political polarization questionnaire data from a CSV or Excel file.

    Args:
        file_path (str): Path to the political polarization data file.
    Returns:
        pd.DataFrame: DataFrame containing the political polarization data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Political polarization file not found: {file_path}")

    if file_path.endswith('.csv'):
        polarization_data = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        polarization_data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    return polarization_data

def load_posts_ratings(root_path):
    """
    Loads social media posts ratings data from txt files.
    Merges all txt files in the specified directory into a single DataFrame in a wide format -
    each row represents a subject, and each column represents a type of rating of a post type (5 X 4 = 20 columns).

    Args:
        root_path (str): Path to the directory containing the posts ratings txt files.
    Returns:
        pd.DataFrame: DataFrame containing the posts ratings data.
    """
    all_files = [f for f in os.listdir(root_path) if f.endswith('.txt')]
    all_data = []

    for file_name in all_files:
        file_path = os.path.join(root_path, file_name)
        subject_data = pd.read_csv(file_path, sep='\t')

        # Extract subject ID from file name
        subject_id = file_name.split('_')[0]  # Assuming file name format is "00XX_posttype_ratings.txt"
        subject_data['subject_id'] = subject_id

        all_data.append(subject_data)

    combined_data = pd.concat(all_data, ignore_index=True)

    # Pivot the data to wide format
    wide_data = combined_data.pivot_table(index='subject_id', 
                                          columns=['post_type', 'rating_type'], 
                                          values='rating_value').reset_index()

    # Flatten MultiIndex columns
    wide_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in wide_data.columns.values]

    return wide_data