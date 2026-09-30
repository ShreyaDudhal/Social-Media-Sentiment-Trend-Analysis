"""
PySpark Data Processing Engine
Executes Big Data exploratory analytics, aggregations, and statistics using Apache Spark DataFrames.
"""

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SparkProcessor")

# Try importing pyspark components
SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logger.warning("PySpark not installed or Java unavailable. Spark engine will operate in fallback mode.")

class SparkAnalyticsProcessor:
    def __init__(self, app_name="SocialMediaSentimentAnalysis"):
        self.app_name = app_name
        self.spark = None
        self.is_active = False
        if SPARK_AVAILABLE:
            try:
                self.spark = SparkSession.builder \
                    .appName(self.app_name) \
                    .master("local[*]") \
                    .config("spark.driver.bindAddress", "127.0.0.1") \
                    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                    .getOrCreate()
                self.spark.sparkContext.setLogLevel("ERROR")
                self.is_active = True
                logger.info("SparkSession successfully initialized in local[*] mode.")
            except Exception as e:
                logger.warning(f"Failed to start SparkSession: {e}. Switching to pandas compatibility engine.")
                self.is_active = False

    def create_spark_df(self, pandas_df: pd.DataFrame):
        """Converts a Pandas DataFrame into a PySpark DataFrame."""
        if not self.is_active or self.spark is None:
            return None
        try:
            # Clean dataframe for PySpark conversion
            clean_df = pandas_df.copy()
            for col in ['user_id', 'post', 'timestamp', 'hashtags', 'mentioned_user']:
                if col in clean_df.columns:
                    clean_df[col] = clean_df[col].astype(str)
            for col in ['likes', 'comments', 'shares', 'engagement']:
                if col in clean_df.columns:
                    clean_df[col] = clean_df[col].astype(int)
            return self.spark.createDataFrame(clean_df)
        except Exception as e:
            logger.error(f"Error converting Pandas DF to Spark DF: {e}")
            return None

    def compute_overview_kpis(self, pandas_df: pd.DataFrame):
        """
        Calculates core KPI metrics using PySpark DataFrame functions if active, else Pandas.
        """
        spark_df = self.create_spark_df(pandas_df)
        if spark_df is not None:
            try:
                # Genuine PySpark operations: count, select, distinct, agg, sum, avg, max
                total_posts = spark_df.count()
                total_users = spark_df.select("user_id").distinct().count()
                
                agg_row = spark_df.agg(
                    F.sum("likes").alias("total_likes"),
                    F.sum("comments").alias("total_comments"),
                    F.sum("shares").alias("total_shares"),
                    F.avg("likes").alias("avg_likes"),
                    F.avg("comments").alias("avg_comments"),
                    F.avg("shares").alias("avg_shares"),
                    F.avg("engagement").alias("avg_engagement"),
                    F.max("likes").alias("max_likes"),
                    F.max("shares").alias("max_shares")
                ).collect()[0]
                
                return {
                    'engine_used': 'PySpark 3.x (Local Cluster)',
                    'total_posts': int(total_posts),
                    'total_users': int(total_users),
                    'total_likes': int(agg_row['total_likes'] or 0),
                    'total_comments': int(agg_row['total_comments'] or 0),
                    'total_shares': int(agg_row['total_shares'] or 0),
                    'avg_likes': float(round(agg_row['avg_likes'] or 0, 2)),
                    'avg_comments': float(round(agg_row['avg_comments'] or 0, 2)),
                    'avg_shares': float(round(agg_row['avg_shares'] or 0, 2)),
                    'avg_engagement': float(round(agg_row['avg_engagement'] or 0, 2)),
                    'max_likes': int(agg_row['max_likes'] or 0),
                    'max_shares': int(agg_row['max_shares'] or 0)
                }
            except Exception as e:
                logger.error(f"Spark aggregate execution error: {e}")

        # Pandas Fallback
        return {
            'engine_used': 'Pandas Analytics Engine',
            'total_posts': len(pandas_df),
            'total_users': pandas_df['user_id'].nunique(),
            'total_likes': int(pandas_df['likes'].sum()),
            'total_comments': int(pandas_df['comments'].sum()),
            'total_shares': int(pandas_df['shares'].sum()),
            'avg_likes': float(round(pandas_df['likes'].mean(), 2)),
            'avg_comments': float(round(pandas_df['comments'].mean(), 2)),
            'avg_shares': float(round(pandas_df['shares'].mean(), 2)),
            'avg_engagement': float(round(pandas_df['engagement'].mean(), 2)),
            'max_likes': int(pandas_df['likes'].max()),
            'max_shares': int(pandas_df['shares'].max())
        }

    def get_most_active_users(self, pandas_df: pd.DataFrame, top_n=10):
        """
        Uses PySpark groupBy, agg, orderBy, withColumn to rank top active users.
        """
        spark_df = self.create_spark_df(pandas_df)
        if spark_df is not None:
            try:
                top_users_spark = spark_df.groupBy("user_id") \
                    .agg(
                        F.count("post").alias("post_count"),
                        F.sum("likes").alias("total_likes"),
                        F.sum("shares").alias("total_shares"),
                        F.avg("engagement").alias("avg_engagement")
                    ) \
                    .orderBy(F.col("post_count").desc(), F.col("total_likes").desc()) \
                    .limit(top_n)
                
                res_df = top_users_spark.toPandas()
                res_df['avg_engagement'] = res_df['avg_engagement'].round(2)
                res_df.insert(0, 'Rank', range(1, len(res_df) + 1))
                return res_df
            except Exception as e:
                logger.error(f"Spark groupBy user error: {e}")

        # Pandas Fallback
        res_df = pandas_df.groupby('user_id').agg(
            post_count=('post', 'count'),
            total_likes=('likes', 'sum'),
            total_shares=('shares', 'sum'),
            avg_engagement=('engagement', 'mean')
        ).reset_index().sort_values(by=['post_count', 'total_likes'], ascending=False).head(top_n)
        res_df['avg_engagement'] = res_df['avg_engagement'].round(2)
        res_df.insert(0, 'Rank', range(1, len(res_df) + 1))
        return res_df

    def get_most_shared_posts(self, pandas_df: pd.DataFrame, top_n=10):
        """
        Uses PySpark select, filter, orderBy to retrieve top shared posts.
        """
        spark_df = self.create_spark_df(pandas_df)
        if spark_df is not None:
            try:
                top_posts_spark = spark_df.select("user_id", "post", "likes", "comments", "shares", "engagement") \
                    .orderBy(F.col("shares").desc(), F.col("likes").desc()) \
                    .limit(top_n)
                res_df = top_posts_spark.toPandas()
                res_df.insert(0, 'Rank', range(1, len(res_df) + 1))
                return res_df
            except Exception as e:
                logger.error(f"Spark top shared post error: {e}")

        # Pandas Fallback
        res_df = pandas_df[['user_id', 'post', 'likes', 'comments', 'shares', 'engagement']] \
            .sort_values(by=['shares', 'likes'], ascending=False).head(top_n)
        res_df.insert(0, 'Rank', range(1, len(res_df) + 1))
        return res_df

    def stop_session(self):
        if self.spark:
            try:
                self.spark.stop()
                self.is_active = False
                logger.info("SparkSession stopped.")
            except Exception:
                pass
