"""
Engagement Analytics Module
Computes post engagement metrics: Likes + Comments + Shares and breakdowns.
"""

import pandas as pd

def compute_engagement_breakdowns(df: pd.DataFrame):
    """
    Computes engagement metrics categorized by sentiment, user activity, and temporal trends.
    """
    if df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
        
    df_temp = df.copy()
    if 'engagement' not in df_temp.columns:
        df_temp['engagement'] = df_temp.get('likes', 0) + df_temp.get('comments', 0) + df_temp.get('shares', 0)
        
    kpis = {
        'avg_engagement': float(round(df_temp['engagement'].mean(), 2)),
        'max_engagement': int(df_temp['engagement'].max()),
        'median_engagement': float(round(df_temp['engagement'].median(), 2)),
        'total_engagement': int(df_temp['engagement'].sum())
    }
    
    # Engagement by Sentiment
    engagement_by_sentiment = pd.DataFrame()
    if 'sentiment' in df_temp.columns:
        engagement_by_sentiment = df_temp.groupby('sentiment').agg(
            total_posts=('engagement', 'count'),
            avg_engagement=('engagement', 'mean'),
            total_likes=('likes', 'sum'),
            total_shares=('shares', 'sum')
        ).reset_index()
        engagement_by_sentiment['avg_engagement'] = engagement_by_sentiment['avg_engagement'].round(2)
        
    # Engagement Over Time
    engagement_over_time = pd.DataFrame()
    if 'timestamp' in df_temp.columns:
        df_temp['date'] = pd.to_datetime(df_temp['timestamp'], errors='coerce').dt.date
        df_temp_clean = df_temp.dropna(subset=['date'])
        engagement_over_time = df_temp_clean.groupby('date').agg(
            daily_posts=('engagement', 'count'),
            daily_engagement=('engagement', 'sum'),
            avg_daily_engagement=('engagement', 'mean')
        ).reset_index()
        engagement_over_time['avg_daily_engagement'] = engagement_over_time['avg_daily_engagement'].round(2)
        
    return kpis, engagement_by_sentiment, engagement_over_time
