# Social Media Sentiment and Trend Analysis Using Big Data Technologies

> **Big Data Analytics Laboratory Mini Project**  
> A web-based Big Data analytics application built with **PySpark**, **Streamlit**, **MongoDB**, **NLTK/VADER**, and **NetworkX**.

---

## 📌 Project Overview
This mini-project demonstrates key Big Data analytics concepts applied to social media datasets. The application allows users to analyze social media posts through:
- **Exploratory Data Analysis (EDA)** using PySpark DataFrames.
- **Sentiment Analysis** using NLTK VADER compound polarity scoring.
- **Hashtag Frequency & Trending Topic Analysis** using dynamic engagement formulas (`Likes + Comments + Shares`).
- **User Activity & Engagement Analytics** tracking top active users and share/retweet distributions.
- **Social Network Analysis (SNA)** constructing user mention graphs and calculating **Degree Centrality**.
- **Community Detection** partitioning user networks with Greedy Modularity algorithms.
- **NoSQL Document Storage** integrating PyMongo for storing processed records in MongoDB (`social_media_db`).
- **Interactive Dashboard** offering 8 analytical tabs built with Streamlit and Plotly.

---

## 🎯 Features & Capabilities
1. **Flexible CSV Preprocessing Layer**: Automatically recognizes column variations (`tweet`/`post`, `retweets`/`shares`, `username`/`user_id`, `created_at`/`timestamp`, `mentions`/`mentioned_user`) and safely handles missing fields without crashing.
2. **PySpark Big Data Analytics Engine**: Runs native Apache Spark DataFrame transformations (`groupBy`, `agg`, `filter`, `select`, `withColumn`, `orderBy`) in `local[*]` mode.
3. **Resilient MongoDB Handler**: Automatically checks connection status to `mongodb://localhost:27017/`. Displays a status badge and continues running seamlessly in local mode if MongoDB is offline.
4. **Interactive Network Graph**: Displays 2D user interaction subgraphs with customizable node sizing (degree centrality) and community color partitioning.
5. **Data Explorer & CSV Exporter**: Allows filtering by sentiment, searching by user/hashtag, and downloading `processed_social_media.csv`.
6. **Built-in Demonstration Dataset**: Ships with `data/sample_social_media.csv` (350 realistic records, 50 users, tech/AI topics, and cluster interactions).

---

## 🏗️ Folder Structure

```
social-media-sentiment-analysis/
│
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Python Dependencies
├── README.md                   # Complete Project Documentation & Syllabus Mapping
├── .env.example                # Environment Variable Template
├── .gitignore                  # Git Ignore Rules
│
├── data/
│   ├── sample_social_media.csv # Demonstration Dataset (350 records)
│   └── README.md               # Dataset Documentation
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Flexible CSV Column Mapping & Preprocessing
│   ├── spark_processor.py      # PySpark DataFrame Processing Engine
│   ├── sentiment_analysis.py   # NLTK / VADER Sentiment Analyzer
│   ├── hashtag_analysis.py     # Hashtag Extraction & Trending Score Engine
│   ├── engagement_analysis.py  # Engagement Metrics & Time-Series Aggregation
│   ├── network_analysis.py     # NetworkX Social Graph & Degree Centrality
│   ├── community_detection.py  # Greedy Modularity Community Detection
│   └── mongodb_handler.py      # PyMongo Document Storage & Fallback Handler
│
├── utils/
│   ├── __init__.py
│   └── helpers.py              # CSV Export & Formatting Utilities
│
└── outputs/
    └── .gitkeep                # Directory for Generated Reports
```

---

## 🛠️ Technology Stack
- **Language**: Python 3.10+
- **Big Data Processing**: Apache Spark 3.x / PySpark
- **NoSQL Database**: MongoDB / PyMongo
- **NLP Sentiment**: NLTK VADER (SentimentIntensityAnalyzer)
- **Graph Analytics**: NetworkX
- **Frontend Dashboard**: Streamlit
- **Visualization**: Plotly Express & Plotly Graph Objects
- **Data Wrangling**: Pandas & NumPy

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python**: Version 3.10 or higher installed.
- **Java JDK**: JDK 8, 11, 17, or 19 installed (required for PySpark local execution).
- **MongoDB (Optional)**: Community Server running locally at `localhost:27017`.

### 1. Clone / Open Directory
Open your terminal in the project root folder:
```bash
cd "social media sentiment and social media trend analysis"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

**Activate Virtual Environment**:
- **Windows (Command Prompt / PowerShell)**:
  ```powershell
  .\venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

Upon launching:
1. Open `http://localhost:8501` in your browser.
2. Select **"Use Sample Dataset"** in the sidebar to immediately load and analyze the included 350 demonstration records.
3. Or select **"Upload CSV Dataset"** to upload any public social media CSV file.

---

## 🍃 MongoDB Integration Setup (Optional)
By default, the application checks for MongoDB at `mongodb://localhost:27017/`.

- **If MongoDB is running**:
  The sidebar badge displays `CONNECTED` and stores analyzed posts in `social_media_db.posts` and summary data in `social_media_db.analysis_results`.
- **If MongoDB is offline**:
  The sidebar badge displays `NOT CONNECTED` with the message:
  `"MongoDB not connected - running in local dataset mode."`
  *The entire application continues functioning smoothly without crashing!*

To configure a custom MongoDB connection URI:
Create a `.env` file from `.env.example`:
```env
MONGODB_URI=mongodb://username:password@localhost:27017/
```

---

## 📊 Analytical Methodology

### 1. Sentiment Analysis
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner):
- **Positive**: `Compound Score >= 0.05`
- **Negative**: `Compound Score <= -0.05`
- **Neutral**: `-0.05 < Compound Score < 0.05`

### 2. Trending Score Formula
Calculates total topic traction using engagement metrics:
$$\text{Trending Score} = \text{Likes} + \text{Comments} + \text{Shares}$$

### 3. Social Network Analysis (SNA)
- **Nodes**: Users (`user_id`).
- **Edges**: Directed mention interactions (`user_id -> mentioned_user`).
- **Degree Centrality**: Calculated via NetworkX:
  $$C_D(v) = \frac{deg(v)}{N - 1}$$
  *Indicates connectedness within the interaction network.*

### 4. Community Detection
Partitioning is computed using NetworkX's **Greedy Modularity Community Detection** algorithm on the undirected interaction graph:
$$Q = \sum_{i} (e_{ii} - a_i^2)$$

---

## 🎓 Mapping to Big Data Analytics Lab Syllabus

| Syllabus Topic | Practical Project Component |
| :--- | :--- |
| **MongoDB / NoSQL** | PyMongo integration storing processed post documents and analytical results in `social_media_db`. |
| **PySpark / Spark** | `SparkSession` executing DataFrame operations (`select`, `filter`, `groupBy`, `agg`, `orderBy`). |
| **Descriptive Analytics** | Aggregated KPI cards, engagement averages, maximum likes/shares, and temporal statistics. |
| **Social Network Analysis** | NetworkX interaction graph construction and Degree Centrality evaluation. |
| **Community Detection** | Greedy Modularity community clustering and colored graph visualizations. |
| **Data Visualization** | Plotly charts (donut, time-series, bar, 2D network graphs) embedded in Streamlit. |
| **Mini Project Implementation** | End-to-end working lab mini-project ready for demonstration. |

---

## 📷 Screenshots Placeholder
- **Overview Dashboard**: Displays KPI cards, sentiment pills, data preview, and summary statistics.
- **Sentiment Tab**: Donut charts, post count comparisons, daily sentiment trends, and score tables.
- **Social Network Tab**: Interactive Plotly user network graph and top degree centrality user table.
- **Community Detection Tab**: Modularity partitioned network graph with community filters.

---

## 🤝 Project Credits
Created for **Big Data Analytics Laboratory Mini Project**.
#   S o c i a l - M e d i a - S e n t i m e n t - T r e n d - A n a l y s i s  
 