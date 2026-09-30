import csv
import random
import os
from datetime import datetime, timedelta

def generate_sample_data():
    os.makedirs("c:/Program Files/Java/jdk-19/bin/practicals/Shreya/social media sentiment and social media trend analysis/data", exist_ok=True)
    file_path = "c:/Program Files/Java/jdk-19/bin/practicals/Shreya/social media sentiment and social media trend analysis/data/sample_social_media.csv"
    
    users = [f"U{i:03d}" for i in range(1, 51)] # U001 to U050
    
    topics = {
        "AI": ["#AI #MachineLearning #DeepLearning", "#AI #ArtificialIntelligence #TechTrends", "#AI #GenAI #FutureOfWork"],
        "Python": ["#Python #DataScience #Coding", "#Python #PySpark #BigData", "#Python #Developer #Programming"],
        "Cloud": ["#CloudComputing #AWS #Azure", "#Cloud #DevOps #BigData", "#Cloud #Tech #CyberSecurity"],
        "Data Science": ["#DataScience #Analytics #BigData", "#DataScience #Statistics #PySpark", "#DataScience #DataVisualisation #Tech"],
        "Education": ["#Education #EdTech #Learning", "#Education #BigDataAnalytics #StudentLife", "#Education #Tech #Career"],
        "Social Media": ["#SocialMedia #Trends #DigitalMarketing", "#SocialMedia #SentimentAnalysis #Analytics", "#SocialMedia #Networking #Tech"]
    }
    
    positive_templates = [
        "Extremely impressed with the new PySpark 3.5 release! The performance gains in big data processing are incredible.",
        "Loving how easy it is to analyze social network graphs using NetworkX and PySpark! Great learning experience.",
        "Attended a fantastic webinar on AI and Big Data Analytics today. Feeling very inspired! cc @{mention}",
        "Python and PySpark make complex data pipelines look effortless! High efficiency all around. @{mention}",
        "MongoDB integration with PyMongo is so smooth. Seamless NoSQL document storage for our mini project!",
        "Big Data Analytics laboratory sessions are super interesting this semester. Thanks to @{mention} for helping out!",
        "Excited about the latest breakthroughs in Generative AI and Large Language Models. What a time to be in tech!",
        "Check out this awesome dashboard built with Streamlit and Plotly. Super clean UI! @{mention}",
        "Cloud computing paired with Apache Spark handles millions of transactions effortlessly. Brilliant architecture!",
        "Just completed our community detection project using Greedy Modularity algorithms! Works like a charm with @{mention}.",
        "Super happy with our sentiment analysis results. VADER sentiment model gave 95% accurate scores on our dataset.",
        "Wonderful workshop on graph analytics and degree centrality metrics by @{mention}. Highly recommended!",
        "Data science and big data analytics are transforming how we understand social media trends. Thrilled to build this!",
        "Big thanks to @{mention} for sharing insightful tips on PySpark DataFrame optimizations!"
    ]
    
    neutral_templates = [
        "Analyzing user activity and post engagement across 10,000 social media records today. @{mention}",
        "Comparing PySpark DataFrame operations vs Pandas for large scale data preprocessing tasks.",
        "Running community detection algorithms on user mention networks. Results look balanced.",
        "Exploring MongoDB collections and NoSQL schema design for social media post storage.",
        "Discussing social network graph metrics including degree centrality and modularity with @{mention}.",
        "Updated the dataset with retweets, likes, and comment counts for trend analysis benchmarking.",
        "Reviewing the syllabus requirements for Big Data Analytics Lab project submission. @{mention}",
        "Submitted the exploratory data analysis report on hashtag frequencies and sentiment distribution.",
        "Testing VADER sentiment compound thresholds on synthetic vs real Twitter dataset records. @{mention}",
        "Configuring local SparkSession master settings local[*] for desktop execution testing.",
        "Scheduled a group meeting with @{mention} to discuss PyMongo connection fallback logic.",
        "Reading papers on social network analysis and community detection algorithms in python."
    ]
    
    negative_templates = [
        "Frustrated with memory errors when running unoptimized Spark joins on massive datasets. Need to refactor @{mention}",
        "MongoDB service failed to start locally. Thankfully our app has a fallback mode, but annoying nonetheless.",
        "Disappointed with noisy data quality in raw social media posts. Preprocessing took way longer than expected.",
        "Too many missing values in the retweets and shares column! Standardizing CSV schemas is painful. @{mention}",
        "Network visualization gets cluttered when plotting over 1000 nodes at once. Need to filter top centrality users.",
        "Spark job failed due to missing dependency drivers on Windows environment. Spent hours debugging with @{mention}.",
        "Struggling with slow network graph rendering on heavy social media datasets. Optimizing layout now.",
        "Corrupted CSV input caused parser exceptions during PySpark read. Adding validation checks immediately @{mention}",
        "Social media API rate limits are extremely restrictive now. Glad we can use custom CSV datasets instead.",
        "VADER sentiment misclassified sarcasm in some negative tweets. Fine-tuning thresholds with @{mention}."
    ]
    
    records = []
    base_time = datetime.now() - timedelta(days=30)
    
    # Ensure distinct network structure with clusters
    clusters = [
        ["U001", "U002", "U003", "U004", "U005", "U006", "U007", "U008"],
        ["U009", "U010", "U011", "U012", "U013", "U014", "U015"],
        ["U016", "U017", "U018", "U019", "U020", "U021", "U022", "U023"],
        ["U024", "U025", "U026", "U027", "U028", "U029", "U030"],
        ["U031", "U032", "U033", "U034", "U035", "U036", "U037", "U038", "U039", "U040"]
    ]
    
    for i in range(350):
        # Pick user
        user = random.choice(users)
        
        # Pick mentioned user (prefer same cluster for strong community detection signal, but occasionally cross-cluster)
        user_cluster = next((c for c in clusters if user in c), users)
        if random.random() < 0.75 and user_cluster:
            mentioned = random.choice([u for u in user_cluster if u != user] or users)
        else:
            mentioned = random.choice([u for u in users if u != user])
            
        # Determine sentiment pool
        rand_val = random.random()
        if rand_val < 0.45:
            template = random.choice(positive_templates)
        elif rand_val < 0.80:
            template = random.choice(neutral_templates)
        else:
            template = random.choice(negative_templates)
            
        post_text = template.format(mention=mentioned)
        
        # Pick topic & hashtags
        topic_name = random.choice(list(topics.keys()))
        hashtag_str = random.choice(topics[topic_name])
        
        # Generate metrics
        likes = random.randint(5, 450)
        comments = random.randint(0, 85)
        shares = random.randint(0, 120)
        
        # Timestamp
        dt = base_time + timedelta(days=random.randint(0, 29), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        records.append({
            "user_id": user,
            "post": post_text,
            "timestamp": timestamp_str,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "hashtags": hashtag_str,
            "mentioned_user": mentioned
        })
        
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "post", "timestamp", "likes", "comments", "shares", "hashtags", "mentioned_user"])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Successfully generated {len(records)} sample records in {file_path}")

if __name__ == "__main__":
    generate_sample_data()
