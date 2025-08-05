# 🎭 Hospital Reviews Topic Modelling and Sentiment Analysis

![python](https://img.shields.io/badge/python-3.11.9-red) ![pandas](https://img.shields.io/badge/pandas-2.2.3-orange) ![nltk](https://img.shields.io/badge/nltk-3.9.1-yellow) ![matplotlib](https://img.shields.io/badge/matplotlib-3.10.0-green) ![seaborn](https://img.shields.io/badge/seaborn-0.13.2-blue) ![wordcloud](https://img.shields.io/badge/wordcloud-1.9.4-indigo) ![scikit_learn](https://img.shields.io/badge/scikit_learn-1.6.1-violet) ![xgboost](https://img.shields.io/badge/xgboost-3.0.2-red) ![scipy](https://img.shields.io/badge/scipy-1.15.2-orange) 

This project applies advanced Natural Language Processing (NLP) and machine learning techniques to analyze hospital reviews collected from Google Maps. The goal is to uncover key themes in patient feedback and accurately classify sentiment, providing actionable insights for healthcare providers and stakeholders.

## Repository Overview

|Description|File|
|:-|:-|
|Analysis notebook with all code used to generate the results, including comments and notes. |  `Hospital_Reviews_Topic_Modelling_and_Sentiment_Analysis.ipynb`|
|Dataset file containing all data used in the analysis. [Hospital Reviews Dataset](https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset) | `hospital.csv`|
|Analysis report discussing findings in more detail. | `Hospital_Reviews_Analysis_Report.pdf`|

## Project Overview

This project demonstrates the use of text analysis techniques to better understand public feedback by analyzing online reviews. It applies **Latent Dirichlet Allocation (LDA)** for topic modelling and **Extreme Gradient Boosting (XGBoost)** to predict sentiment labels.

- **Dataset:** [Hospital Reviews Dataset](https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset)  
  996 sentiment-labelled reviews from hospitals in Bengaluru, India, including both written feedback and 1–5 star ratings.

- **Objectives:**
  - **Topic Modelling:** Identify major themes and areas of concern in patient feedback using Latent Dirichlet Allocation (LDA).
  - **Sentiment Analysis:** Build robust classifiers (XGBoost) to predict review sentiment, addressing class imbalance and leveraging both text and ratings.

## Key Features

- **Data Cleaning & Preprocessing:**  
  - Removal of duplicates and irrelevant columns  
  - Text normalization (lowercasing, punctuation removal, tokenization, stopword filtering, lemmatization with NLTK)

- **Exploratory Data Analysis (EDA):**  
  - Visualizations of sentiment and rating distributions  
  - Word clouds for quick thematic inspection

- **Topic Modelling:**  
  - LDA to extract interpretable topics from cleaned feedback  
  - Topic assignment for each review and summary statistics per topic

- **Sentiment Classification:**  
  - XGBoost models trained on TF-IDF features, with and without class weighting  
  - Integration of numerical ratings as features for improved accuracy  
  - Evaluation using accuracy, precision, recall, F1, ROC AUC, and confusion matrices

- **Result Interpretation:**  
  - Topic-level sentiment and rating summaries  
  - Visualizations of model performance by topic

## Results

- **Topic Modelling:**  
  Four clear themes emerged:  
  1. Frustration & Poor Experience  
  2. Exceptional Clinical Care  
  3. General Satisfaction with Hospital  
  4. Communication & Process Issues

- **Sentiment Analysis:**  
  - Text-only models performed well but struggled with minority (negative) class due to imbalance.
  - Class weighting improved fairness, especially for negative feedback.
  - Including ratings as a feature led to near-perfect sentiment prediction, highlighting the strong correlation between ratings and sentiment labels.

## Technologies Used

- Python (Pandas, NumPy, scikit-learn, XGBoost, NLTK, Matplotlib, Seaborn, WordCloud)
- Jupyter Notebook

## Insights & Impact

- **Actionable Insights:**  
  - Pinpointed key drivers of patient satisfaction and dissatisfaction.
  - Provided a framework for healthcare providers to monitor and improve service quality.

- **Methodological Value:**  
  - Demonstrates the power of combining structured (ratings) and unstructured (text) data.
  - Shows the importance of addressing class imbalance in real-world sentiment analysis.



**Author:** Jishen Harilal  
**LinkedIn:** www.linkedin.com/in/jishen-harilal  
**Contact:** jishen2108@gmail.com  

---

*For more details, see the full analysis in Hospital_Reviews_Topic_Modelling_and_Sentiment_Analysis.ipynb.*
