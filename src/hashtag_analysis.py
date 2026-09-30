"""
Hashtag Frequency and Trending Topic Analysis Module
Calculates hashtag occurrences and computes Trending Score = Likes + Comments + Shares
"""

import pandas as pd
from collections import Counter
from src.data_preprocessing import extract_hashtags_list


def analyze_hashtags_and_trends(df: pd.DataFrame, top_n=10):
    """
    Extracts, normalizes, and ranks hashtags by frequency and engagement score.
    Trending Score Formula:
        Trending Score = Likes + Comments + Shares
    Returns:
        pd.DataFrame: Top N hashtags with post_count, total_likes, total_shares, trending_score.
        dict: Hashtag analysis summary metrics.
    """
    if df.empty or 'hashtags' not in df.columns:
        return pd.DataFrame(), {'total_hashtags': 0, 'unique_hashtags': 0}
        
    hashtag_data = {}
    total_tag_instances = 0
    
    for idx, row in df.iterrows():
        tags = extract_hashtags_list(row['hashtags'])
        if not tags and 'post' in df.columns:
            tags = extract_hashtags_list(row['post'])
            
        likes = row.get('likes', 0)
        comments = row.get('comments', 0)
        shares = row.get('shares', 0)
        engagement = likes + comments + shares
        
        for tag in set(tags): # Unique tags per post
            tag_name = f"#{tag}"
            total_tag_instances += 1
            if tag_name not in hashtag_data:
                hashtag_data[tag_name] = {
                    'post_count': 0,
                    'total_likes': 0,
                    'total_comments': 0,
                    'total_shares': 0,
                    'trending_score': 0
                }
                
            hashtag_data[tag_name]['post_count'] += 1
            hashtag_data[tag_name]['total_likes'] += likes
            hashtag_data[tag_name]['total_comments'] += comments
            hashtag_data[tag_name]['total_shares'] += shares
            hashtag_data[tag_name]['trending_score'] += engagement
            
    if not hashtag_data:
        return pd.DataFrame(), {'total_hashtags': 0, 'unique_hashtags': 0}
        
    trend_list = []
    for tag, stats in hashtag_data.items():
        trend_list.append({
            'Hashtag': tag,
            'Posts': stats['post_count'],
            'Likes': stats['total_likes'],
            'Comments': stats['total_comments'],
            'Shares': stats['total_shares'],
            'Trending Score': stats['trending_score']
        })
        
    trend_df = pd.DataFrame(trend_list)
    trend_df = trend_df.sort_values(by=['Trending Score', 'Posts'], ascending=False).reset_index(drop=True)
    trend_df.insert(0, 'Rank', range(1, len(trend_df) + 1))
    
    summary = {
        'total_hashtag_instances': total_tag_instances,
        'unique_hashtags': len(hashtag_data),
        'top_trending_tag': trend_df.iloc[0]['Hashtag'] if not trend_df.empty else 'N/A'
    }
    
    return trend_df.head(top_n), summary
