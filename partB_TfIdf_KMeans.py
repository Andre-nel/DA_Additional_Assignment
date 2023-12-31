import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
Segmentation of leaf spots disease in apple plants using particle swarm optimization and K-means algorithm; agriculture"""

# Define a regular expression pattern to extract information
pattern = r'(?P<Title>.*?); (?P<Category>.*?)$'

# Create an empty list to store the extracted data
data = []

# Split the text into lines and extract information using regex
for line in text.strip().split('\n'):
    match = re.search(pattern, line)
    if match:
        data.append(match.groupdict())

# Create a pandas DataFrame from the extracted data
df = pd.DataFrame(data)

# Preprocess the text
def preprocess_text(text: str, remove_stopwords: bool) -> str:
    text = re.sub(r"http\S+", "", text)  # Remove links
    text = re.sub("[^A-Za-z]+", " ", text)  # Remove special characters and numbers
    if remove_stopwords:
        tokens = nltk.word_tokenize(text)  # Tokenize
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]  # Remove stopwords
        text = " ".join(tokens)  # Join tokens
    text = text.lower().strip()  # Convert to lowercase and remove whitespace
    return text

df['cleaned'] = df['Title'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95)

# Fit and transform the text data
X = vectorizer.fit_transform(df['cleaned'])

# Initialize KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
kmeans.fit(X)

# Store cluster labels in a variable
clusters = kmeans.labels_

# Initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)

# Transform the TF-IDF vectors using PCA
pca_vecs = pca.fit_transform(X.toarray())

# Save PCA dimensions
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

# Add cluster and PCA dimensions to the DataFrame
df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1

# Define a mapping for cluster labels
cluster_map = {0: "technical", 1: "agriculture", 2: "medicine"}

# Apply the mapping to cluster labels
df['cluster'] = df['cluster'].map(cluster_map)

# Set image size for the scatter plot
plt.figure(figsize=(12, 7))

# Set title and axis labels
plt.title("TF-IDF + KMeans Clustering", fontdict={"fontsize": 18})
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})

# Create a scatter plot with cluster coloring
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="deep")
plt.show()

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Extract true cluster labels and assigned clusters
true_labels = df['Category']
assigned_clusters = df['cluster']

# Encode true cluster labels
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Encode assigned clusters
assigned_clusters_encoded = label_encoder.transform(assigned_clusters)

# Compute evaluation metrics
ari = adjusted_rand_score(true_labels_encoded, assigned_clusters_encoded)
nmi = normalized_mutual_info_score(true_labels_encoded, assigned_clusters_encoded)
silhouette = silhouette_score(X, assigned_clusters)

# Print the evaluation metrics
print(f"Adjusted Rand Index (ARI): {ari}")
print(f"Normalized Mutual Information (NMI): {nmi}")
print(f"Silhouette Score: {silhouette}")
