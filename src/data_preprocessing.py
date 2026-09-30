"""
Data Preprocessing & Flexible Column Mapping Layer
Handles column name variations in public Twitter/Social-Media CSV datasets.
Disables missing features gracefully without crashing.
"""

import pandas as pd
import numpy as np
import re
import ast

COLUMN_MAPPINGS = {
    'post': ['post', 'tweet', 'text', 'content', 'message', 'status', 'full_text'],
    'user_id': ['user_id', 'username', 'user', 'author', 'screen_name', 'sender', 'user_name'],
    'timestamp': ['timestamp', 'created_at', 'date', 'time', 'datetime', 'post_date', 'created_time'],
    'likes': ['likes', 'like_count', 'favorites', 'favorite_count', 'upvotes'],
    'comments': ['comments', 'comment_count', 'replies', 'reply_count'],
    'shares': ['shares', 'retweets', 'retweet_count', 'share_count', 'reposts'],
    'hashtags': ['hashtags', 'hashtag', 'tags', 'tag', 'topics'],
    'mentioned_user': ['mentioned_user', 'mentions', 'mentioned_users', 'target_user', 'mention', 'user_mentions']
}

def map_and_preprocess_columns(df: pd.DataFrame):
    """
    Normalizes column names, handles missing data safely, and extracts structured fields.
    Returns:
        pd.DataFrame: Cleaned dataframe with standardized column names.
        dict: Metadata dictionary highlighting missing columns & warnings.
    """
    df = df.copy()
    original_cols = [str(c).strip().lower() for c in df.columns]
    col_rename_dict = {}
    missing_cols = []
    found_targets = set()

    for target_col, synonyms in COLUMN_MAPPINGS.items():
        matched = None
        for col in df.columns:
            clean_col = str(col).strip().lower()
            if clean_col in synonyms:
                matched = col
                break
        if matched:
            col_rename_dict[matched] = target_col
            found_targets.add(target_col)
        else:
            missing_cols.append(target_col)

    # Apply column renaming
    df = df.rename(columns=col_rename_dict)

    # Process missing required columns safely with default fallback values
    if 'post' not in df.columns:
        df['post'] = "No text content available"
    
    if 'user_id' not in df.columns:
        df['user_id'] = "anonymous_user"

    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime("%Y-%m-%d %H:%M:%S").fillna(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    for numeric_col in ['likes', 'comments', 'shares']:
        if numeric_col not in df.columns:
            df[numeric_col] = 0
        else:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce').fillna(0).astype(int)
            df[numeric_col] = df[numeric_col].apply(lambda x: max(0, x)) # Ensure no negative values

    if 'hashtags' not in df.columns:
        df['hashtags'] = ""
    else:
        df['hashtags'] = df['hashtags'].fillna("").astype(str)

    if 'mentioned_user' not in df.columns:
        df['mentioned_user'] = ""
    else:
        df['mentioned_user'] = df['mentioned_user'].fillna("").astype(str)

    # Compute Engagement
    df['engagement'] = df['likes'] + df['comments'] + df['shares']

    # Preprocessing status report
    status_report = {
        'total_rows': len(df),
        'renamed_columns': col_rename_dict,
        'missing_columns': missing_cols,
        'has_mentions': 'mentioned_user' in found_targets or (df['mentioned_user'].str.len() > 0).any(),
        'has_shares': 'shares' in found_targets,
        'has_hashtags': 'hashtags' in found_targets or (df['hashtags'].str.len() > 0).any()
    }

    return df, status_report

def extract_hashtags_list(text_or_hashtags):
    """
    Parses hashtags from varying formats: "#AI #Python", "['AI', 'Python']", "AI, Python"
    Returns a clean list of lowercased hashtag strings.
    """
    if not text_or_hashtags or pd.isna(text_or_hashtags):
        return []
    
    val_str = str(text_or_hashtags).strip()
    
    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                return [str(tag).strip('#').strip().lower() for tag in parsed if str(tag).strip()]
        except Exception:
            pass
            
    # Regex find words starting with #
    tags = re.findall(r'#(\w+)', val_str)
    if tags:
        return [t.lower() for t in tags]
        
    # Split by comma or space if no '#' present
    items = [re.sub(r'[^\w]', '', word).lower() for word in val_str.replace(',', ' ').split()]
    return [i for i in items if i]

def extract_mentions_list(text_or_mentions):
    """
    Parses user mentions from varying formats.
    """
    if not text_or_mentions or pd.isna(text_or_mentions):
        return []
    val_str = str(text_or_mentions).strip()
    
    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                return [str(m).strip('@').strip() for m in parsed if str(m).strip()]
        except Exception:
            pass
            
    mentions = re.findall(r'@(\w+)', val_str)
    if mentions:
        return mentions
        
    items = [re.sub(r'[^\w]', '', word) for word in val_str.replace(',', ' ').split()]
    return [i for i in items if i and i.lower() != 'nan']
