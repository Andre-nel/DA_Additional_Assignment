import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

# Load NLTK stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

# Sample text data
text = """
Research on K-value selection method of K-means clustering algorithm; technical
Performance of K-means clustering algorithm with different distance metrics; technical
Introduction to the K-means clustering algorithm based on the elbow method; technical
Effect of distance metrics in determining k-value in K-means clustering using elbow and silhouette method; technical
K-means clustering with incomplete data; technical
Use of latent class analysis and K-means clustering to identify complex patient profiles; medicine
Covid-19 cases and deaths in southeast Asia clustering using K-means algorithm; medicine
K-means clustering of Covid-19 cases in Indonesia's provinces; medicine
Brain tumor segmentation using K-means clustering and deep learning with synthetic data augmentation for classification; medicine
Skin cancer detection from dermoscopic images using deep learning and fuzzy K-means clustering; medicine
Diagnosis of grape leaf diseases using automatic K-means clustering and machine learning; agriculture
K-means algorithm for clustering system of plant seeds specialization areas in east Aceh; agriculture
Plant disease detection and recognition using K-means clustering; agriculture
Plant leaf recognition using texture features and semi-supervised spherical K-means clustering; agriculture
Segmentation of leaf spots disease in apple plants using particle swarm opti-mization and K-means algorithm; agriculture"""

# Define a function to extract title and category
def extract_title_category(text):
    pattern = r'(?P<Title>.*?); (?P<Category>.*?)$'
    match = re.search(pattern, text)
    if match:
        return match.group('Title'), match.group('Category')
    else:
        return None, None

# Extract titles and categories
data = [extract_title_category(line) for line in text.strip().split('\n')]

# Create a DataFrame
df = pd.DataFrame(data, columns=['Title', 'Category'])

# Define a custom text preprocessing transformer
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i, text in enumerate(X_copy):
            # Remove links
            text = re.sub(r"http\S+", "", text)
            # Remove special characters and numbers
            text = re.sub("[^A-Za-z]+", " ", text)
            # Remove stopwords if specified
            if self.remove_stopwords:
                tokens = nltk.word_tokenize(text)
                tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
                text = " ".join(tokens)
            # Convert to lowercase and strip whitespaces
            text = text.lower().strip()
            X_copy[i] = text
        return X_copy

# Create a pipeline for text preprocessing and vectorization
preprocessing_pipeline = Pipeline([
    ('text_preprocessor', TextPreprocessor(remove_stopwords=True)),
    ('tfidf_vectorizer', TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95))
])

# Fit and transform the text data
X_tfidf = preprocessing_pipeline.fit_transform(df['Title'])

# Use GridSearchCV to find the optimal number of clusters (K)
param_grid = {
    'n_clusters': range(2, 11)
}

kmeans = KMeans(random_state=42)
grid_search = GridSearchCV(kmeans, param_grid, cv=5, scoring='precision', n_jobs=-1)
grid_search.fit(X_tfidf)

# Get the best K from the grid search
best_k = grid_search.best_params_['n_clusters']

# Fit KMeans with the best K
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_tfidf)
clusters = kmeans.labels_

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Apply PCA for visualization
pca = PCA(n_components=2, random_state=42)
pca_vecs = pca.fit_transform(X_tfidf.toarray())
df['PCA1'] = pca_vecs[:, 0]
df['PCA2'] = pca_vecs[:, 1]

# Create a function to get top keywords for each cluster
def get_top_keywords(cluster_centers, terms, n_terms=10):
    top_keywords = []
    for cluster in cluster_centers:
        cluster_terms = [terms[i] for i in cluster.argsort()[:-n_terms-1:-1]]
        top_keywords.append(cluster_terms)
    return top_keywords

# Get cluster centers and feature names
cluster_centers = kmeans.cluster_centers_
feature_names = preprocessing_pipeline.named_steps['tfidf_vectorizer'].get_feature_names_out()

# Get top keywords for each cluster
top_keywords = get_top_keywords(cluster_centers, feature_names)

# Map cluster labels to category names
cluster_map = {0: "technical", 1: "agriculture", 2: "medicine", 3: "other"}  # Adjust as needed
df['Category'] = df['Cluster'].map(cluster_map)

# Visualize the clusters
plt.figure(figsize=(12, 7))
plt.title("TF-IDF + KMeans Clustering", fontsize=18)
plt.xlabel("PCA1", fontsize=16)
plt.ylabel("PCA2", fontsize=16)
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Category', palette="viridis")
plt.show()

# Encode true cluster labels (Category column)
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(df['Category'])

# Compute evaluation metrics
ari = adjusted_rand_score(true_labels_encoded, clusters)
nmi = normalized_mutual_info_score(true_labels_encoded, clusters)
silhouette = silhouette_score(X_tfidf, clusters)

print(f"Adjusted Rand Index (ARI): {ari}")
print(f"Normalized Mutual Information (NMI): {nmi}")
print(f"Silhouette Score: {silhouette}")
