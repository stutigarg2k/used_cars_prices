"""
Reusable data cleaning functions for the used-car price prediction project.

All functions here are deterministic (unit parsing, dropping a known-bad row,
domain-knowledge decisions) rather than statistical (no means/medians/modes
learned from the data). That's what makes it safe to run on the whole dataset
before the train/test split — nothing here could leak train-only information
into a test set. Imputation and encoding are NOT here; those depend on
training-set statistics and belong in the preprocessing Pipeline, built after
the split.
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV as-is, no modifications."""
    return pd.read_csv(path)


def drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the leftover row-index column ('Unnamed: 0').
    EDA confirmed ~0 correlation with Price before this decision was made.
    """
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows. Deterministic, safe pre-split."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"Dropped {removed} duplicate row(s).")
    return df


def remove_implausible_kilometers(df: pd.DataFrame, threshold: int = 1_000_000) -> pd.DataFrame:
    """
    Remove rows with a physically implausible Kilometers_Driven value.
    EDA found one row at 6.5M km (data-entry error); threshold is set well
    above the highest legitimate value found (~775K km) and well below the
    corrupted one, so this removes only the bad row(s), not real high-mileage
    fleet/commercial cars.
    """
    df = df.copy()
    before = len(df)
    df = df[df['Kilometers_Driven'] < threshold].reset_index(drop=True)
    removed = before - len(df)
    print(f"Removed {removed} row(s) with Kilometers_Driven >= {threshold:,}.")
    return df


def parse_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'Engine' from '1582 CC' style strings to numeric CC."""
    df = df.copy()
    df['Engine'] = df['Engine'].str.replace(' CC', '', regex=False)
    df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')
    return df


def parse_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'Power' from '126.2 bhp' style strings to numeric bhp.
    Uses errors='coerce' because some rows contain the literal string
    'null bhp' rather than a true missing value.
    """
    df = df.copy()
    df['Power'] = df['Power'].str.replace(' bhp', '', regex=False)
    df['Power'] = pd.to_numeric(df['Power'], errors='coerce')
    return df


def parse_mileage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split 'Mileage' into a numeric value and its unit.
    kmpl (petrol/diesel) and km/kg (CNG/LPG) are not directly comparable
    quantities, so they're kept as separate columns rather than merged
    into one number.
    """
    df = df.copy()
    split = df['Mileage'].str.split(' ', expand=True)
    df['Mileage_value'] = pd.to_numeric(split[0], errors='coerce')
    df['Mileage_unit'] = split[1]
    df = df.drop(columns=['Mileage'])
    return df


def resolve_new_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'New_Price' (>85% missing, not credibly imputable) with a
    binary 'Had_New_Price' flag, then drop the raw column.
    """
    df = df.copy()
    df['Had_New_Price'] = df['New_Price'].notnull().astype(int)
    df = df.drop(columns=['New_Price'])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full structural cleaning sequence, in order, on a raw DataFrame.
    Order matters: duplicates/outlier removal happen before unit parsing so
    we're not wasting parsing work on rows we're about to drop anyway.
    """
    df = drop_index_column(df)
    df = drop_duplicate_rows(df)
    df = remove_implausible_kilometers(df)
    df = parse_engine(df)
    df = parse_power(df)
    df = parse_mileage(df)
    df = resolve_new_price(df)
    return df