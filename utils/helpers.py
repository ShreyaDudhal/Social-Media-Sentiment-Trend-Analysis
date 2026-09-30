"""
Helper Utilities Module
Provides formatted string exports, CSV dataset converters, and dashboard UI styling helpers.
"""

import pandas as pd

def convert_df_to_csv(df: pd.DataFrame):
    """Encodes a DataFrame into a CSV byte string for Streamlit download button."""
    return df.to_csv(index=False).encode('utf-8')

def format_number(val):
    """Formats large numbers nicely (e.g. 1.2K)."""
    if val is None:
        return "0"
    if val >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val/1_000:.1f}K"
    return str(val)
