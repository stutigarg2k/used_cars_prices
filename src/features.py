"""
Feature engineering functions for the used-car price prediction project.

Like data_prep.py, everything here is deterministic — string splitting and
simple arithmetic, not statistics learned from the data — so it's safe to run
on the whole dataset before the train/test split. Frequency-based grouping of
rare Model categories is deliberately NOT done here; it needs to be fit on the
training split only, so it belongs in the preprocessing pipeline.
"""

import pandas as pd
import numpy as np


def extract_brand_and_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the free-text 'Name' column into 'Brand' (first word) and
    'Model' (everything after it). Brand is low-cardinality and usable
    directly; Model is high-cardinality and will need grouping/encoding
    decisions made later, in the preprocessing pipeline.
    """
    df = df.copy()
    df['Brand'] = df['Name'].str.split(' ').str[0]
    df['Model'] = df['Name'].str.split(' ', n=1).str[1]
    return df


def add_car_age(df: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    """
    Add 'Car_Age' = reference_year - Year.

    reference_year should be the year the data was collected/scraped, NOT
    the current real-world year. Using today's date would make Car_Age drift
    over time and mean something different every time this notebook is re-run
    — using a fixed reference year keeps the feature reproducible and tied to
    when these listings were actually posted.
    """
    df = df.copy()
    df['Car_Age'] = reference_year - df['Year']
    return df


def engineer_features(df: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    """Run the full feature engineering sequence, in order."""
    df = extract_brand_and_model(df)
    df = add_car_age(df, reference_year)
    return df

def add_log_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Price_log' = log1p(Price). Price is heavily right-skewed (EDA);
    log1p is close to symmetric and is the target actually trained on.
    Raw 'Price' is kept alongside it for interpretable error reporting later
    (predictions get np.expm1()'d back before computing/reporting metrics).
    """
    df = df.copy()
    df['Price_log'] = np.log1p(df['Price'])
    return df