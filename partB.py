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


# Read the CSV file
data = pd.read_csv('C:/Users/Jakne/Desktop/Andre/DA_Additional_Assignment/arxiv2017.csv', delimiter=';', nrows=5000)

# Add the 'Title' column to the 'Abstract' column and store it in a new column 'combined_text'
data['combined_text'] = data['Title'] + ' ' + data['Abstract']


def preprocess_data(columnToClean: str, cleanedColumnName: str, selected_subjects: list) -> pd.DataFrame:
    """
    Filter it by selected subjects, and preprocess the text.
    
    Args:
        selected_subjects (list): List of subject areas to filter the data.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Filter data by selected subjects
    filtered_data = data[data['Subject_area'].isin(selected_subjects)]

    # Preprocess the text
    def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
        """
        Preprocess text by removing links, special characters, numbers, and optionally stopwords.
        
        Args:
            text (str): Input text.
            remove_stopwords (bool): Whether to remove stopwords (default True).
            
        Returns:
            str: Preprocessed text.
        """
        text = re.sub(r"http\S+", "", text)  # Remove links
        text = re.sub("[^A-Za-z]+", " ", text)  # Remove special characters and numbers
        if remove_stopwords:
            tokens = nltk.word_tokenize(text)  # Tokenize
            tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]  # Remove stopwords
            text = " ".join(tokens)  # Join tokens
        text = text.lower().strip()  # Convert to lowercase and remove whitespace
        return text

    # Apply text preprocessing to the column to clean
    filtered_data[cleanedColumnName] = filtered_data[columnToClean].apply(
        lambda x: preprocess_text(x, remove_stopwords=True))

    # Save the filtered DataFrame to a CSV file
    filtered_data.to_csv('filtered_data.csv', index=False)

    return filtered_data

# List of selected subjects
selected_subjects = ["DB", "NI", "CR", "CV"]
columnToClean='combined_text'
cleanedColumnName='cleaned'
numClusters = len(selected_subjects)

# Load and preprocess data
# filtered_data = preprocess_data(columnToClean=columnToClean,
#                                 cleanedColumnName=cleanedColumnName,
#                                 selected_subjects=selected_subjects)

filtered_data = pd.read_csv('C:/Users/Jakne/Desktop/Andre/DA_Additional_Assignment/filtered_data.csv')

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(filtered_data['Subject_area'])

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95)

X_tfidf = vectorizer.fit_transform(filtered_data[cleanedColumnName])

from utils import fit_and_evaluate

kmeans = KMeans(
    n_clusters=numClusters,
    max_iter=100,
    n_init=5,
)

fit_and_evaluate(kmeans, X_tfidf, labels=true_labels,
                 name="KMeans\non tf-idf vectors")

a = 1

"""
clustering done in 0.17 ± 0.11 s
Homogeneity: 0.538 ± 0.014
Completeness: 0.530 ± 0.011
V-measure: 0.534 ± 0.012
Adjusted Rand-Index: 0.479 ± 0.022
Normalized Mutual Info: 0.534 ± 0.012
Silhouette Coefficient: 0.008 ± 0.000
"""

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time

lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
X_lsa = lsa.fit_transform(X_tfidf)
explained_variance = lsa[0].explained_variance_ratio_.sum()

print(f"LSA done in {time() - t0:.3f} s")
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

kmeans = KMeans(
    n_clusters=numClusters,
    max_iter=100,
    n_init=1,
)

fit_and_evaluate(kmeans, X_lsa, labels=true_labels,
                 name="KMeans\nwith LSA on tf-idf vectors")
"""
clustering done in 0.02 ± 0.00 s 
Homogeneity: 0.560 ± 0.019
Completeness: 0.546 ± 0.022
V-measure: 0.553 ± 0.020
Adjusted Rand-Index: 0.522 ± 0.025
Normalized Mutual Info: 0.553 ± 0.020
Silhouette Coefficient: 0.039 ± 0.000
"""

original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(numClusters):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :10]:
        print(f"{terms[ind]} ", end="")
    print()

"""
Cluster 0: wireless network networks mobile performance access throughput communication nodes paper
Cluster 1: data security web based query privacy xml paper information users
Cluster 2: image images learning deep method visual training data object using
Cluster 3: sensor wireless networks energy wsns nodes network wsn routing protocol
"""