"""
NLTK / VADER Sentiment Analysis Module
Performs sentiment scoring (positive, negative, neutral, compound score) on social media text.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("SentimentAnalyzer")

# Try loading VADER sentiment analyzer
VADER_AVAILABLE = False
analyzer = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            analyzer = SentimentIntensityAnalyzer()
            VADER_AVAILABLE = True
        except Exception:
            nltk.download('vader_lexicon', quiet=True)
            analyzer = SentimentIntensityAnalyzer()
            VADER_AVAILABLE = True
    except Exception as e:
        logger.warning(f"VADER Sentiment library not found: {e}. Simple lexicon fallback will be used.")
        VADER_AVAILABLE = False


def rule_based_fallback_sentiment(text: str):
    """Simple rule-based fallback if VADER library is absent."""
    pos_words = {'love', 'great', 'awesome', 'impressive', 'good', 'fantastic', 'excellent', 'inspired', 'happy', 'smooth', 'seamless', 'transformation', 'brilliant'}
    neg_words = {'frustrated', 'disappointed', 'failed', 'annoying', 'error', 'corrupted', 'painful', 'cluttered', 'struggling', 'bad', 'horrible', 'restrictive'}
    
    tokens = set(str(text).lower().split())
    pos_count = len(tokens.intersection(pos_words))
    neg_count = len(tokens.intersection(neg_words))
    
    total = pos_count + neg_count
    if total == 0:
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0, 'sentiment': 'Neutral'}
    
    compound = (pos_count - neg_count) / max(1, total)
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return {
        'neg': float(neg_count / max(1, total)),
        'neu': float(1.0 - (pos_count + neg_count)/max(1, total)),
        'pos': float(pos_count / max(1, total)),
        'compound': float(compound),
        'sentiment': sentiment
    }


def analyze_sentiment(pandas_df: pd.DataFrame, text_column='post'):
    """
    Computes sentiment metrics for each row in the dataframe.
    Adds 'sentiment', 'compound_score', 'pos_score', 'neg_score', 'neu_score'.
    Returns updated DataFrame and sentiment summary dict.
    """
    df = pandas_df.copy()
    
    sentiments = []
    compounds = []
    pos_scores = []
    neg_scores = []
    neu_scores = []
    
    for text in df[text_column]:
        if VADER_AVAILABLE and analyzer is not None:
            scores = analyzer.polarity_scores(str(text))
            compound = scores['compound']
            if compound >= 0.05:
                label = 'Positive'
            elif compound <= -0.05:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            sentiments.append(label)
            compounds.append(round(compound, 3))
            pos_scores.append(round(scores['pos'], 3))
            neg_scores.append(round(scores['neg'], 3))
            neu_scores.append(round(scores['neu'], 3))
        else:
            fb = rule_based_fallback_sentiment(text)
            sentiments.append(fb['sentiment'])
            compounds.append(fb['compound'])
            pos_scores.append(fb['pos'])
            neg_scores.append(fb['neg'])
            neu_scores.append(fb['neu'])
            
    df['sentiment'] = sentiments
    df['compound_score'] = compounds
    df['pos_score'] = pos_scores
    df['neg_score'] = neg_scores
    df['neu_score'] = neu_scores
    
    # Compute summary breakdown
    total = len(df)
    counts = df['sentiment'].value_counts().to_dict()
    pos_count = counts.get('Positive', 0)
    neg_count = counts.get('Negative', 0)
    neu_count = counts.get('Neutral', 0)
    
    summary = {
        'total_posts': total,
        'positive_count': pos_count,
        'neutral_count': neu_count,
        'negative_count': neg_count,
        'positive_pct': round((pos_count / max(1, total)) * 100, 2),
        'neutral_pct': round((neu_count / max(1, total)) * 100, 2),
        'negative_pct': round((neg_count / max(1, total)) * 100, 2),
        'avg_compound': round(df['compound_score'].mean(), 3)
    }
    
    return df, summary


def get_sentiment_over_time(df: pd.DataFrame):
    """
    Groups posts by timestamp date and calculates daily sentiment breakdown.
    """
    if 'timestamp' not in df.columns or df.empty:
        return pd.DataFrame()
        
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['timestamp'], errors='coerce').dt.date
    df_temp = df_temp.dropna(subset=['date'])
    
    grouped = df_temp.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    for col in ['Positive', 'Neutral', 'Negative']:
        if col not in grouped.columns:
            grouped[col] = 0
            
    grouped = grouped.reset_index()
    return grouped
