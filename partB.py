import numpy as np
from time import time
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from utils import (fit_and_evaluate_km, fitAndEvaluateGM, printGMClusterTerms,
                   findOptimalClusters, findOptimalClustersGMM,
                   findOptimalClustersKMeans)
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

# Load stopwords once
nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = set(stopwords.words('english'))


# Read the CSV file
data = pd.read_csv(
    'C:/Users/Jakne/Desktop/Andre/DA_Additional_Assignment/arxiv2017.csv', delimiter=';', nrows=5000)

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
        # Lemmentization did not improve the results

        text = re.sub(r"http\S+", "", text)  # Remove links
        # text = re.sub("[^A-Za-z]+", " ", text)  # Remove special characters and numbers
        if remove_stopwords:
            tokens = nltk.word_tokenize(text)  # Tokenize
            tokens = [w for w in tokens if not w.lower(
            ) in stopwords.words("english")]  # Remove stopwords
            text = " ".join(tokens)  # Join tokens
        text = text.lower().strip()  # Convert to lowercase and remove whitespace
        return text

    # Apply text preprocessing to the column to clean
    filtered_data[cleanedColumnName] = filtered_data[columnToClean].apply(
        lambda x: preprocess_text(x, remove_stopwords=True))

    # Save the filtered DataFrame to a CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"filtered_data_5_clusters_{timestamp}.csv"

    filtered_data.to_csv(filename, index=False)

    return filtered_data


# List of selected subjects
selected_subjects = ["DB", "NI", "CR", "CV", "IT"]    # , "CR", "CV", "IT"
columnToClean = 'combined_text'
cleanedColumnName = 'cleaned'
numClusters = len(selected_subjects)

# Preprocess data
# filtered_data = preprocess_data(columnToClean=columnToClean,
#                                 cleanedColumnName=cleanedColumnName,
#                                 selected_subjects=selected_subjects)

filtered_data = pd.read_csv(
    'C:/Users/Jakne/Desktop/Andre/filtered_data_5_clusters_2023-12-29_18-58-23.csv')


label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(filtered_data['Subject_area'])

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.80)

X_tfidf = vectorizer.fit_transform(filtered_data[cleanedColumnName])


lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
X_lsa = lsa.fit_transform(X_tfidf)
explained_variance = lsa[0].explained_variance_ratio_.sum()

print(f"LSA done in {time() - t0:.3f} s")
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

# findOptimalClustersKMeans(X_lsa, maxClusters=10)

# findOptimalClustersGMM(X_lsa, max_clusters=10)

# todo use gridsearchCV
# optimal_clusters = findOptimalClusters(X_lsa, max_clusters=10, nRuns=7)

gm = GaussianMixture(
    n_components=numClusters,
    max_iter=100,
    n_init=30,
)


gm.fit(X_lsa)

# Assuming 'gm' is your fitted GaussianMixture model and 'X' is your data
probabilities = gm.predict_proba(X_lsa)

"""# def plot_cluster_heatmap(probabilities):
    # 
    # Plot a heatmap of document-cluster probabilities.

    # Args:
    #     probabilities (np.ndarray): Probabilities of documents belonging to each cluster.
    # 
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(probabilities, cmap='viridis')
    # plt.title("Heatmap of Document-Cluster Probabilities")
    # plt.ylabel("Document Index")
    # plt.xlabel("Cluster Index")
    # plt.show()

# probabilities: 1773 x 5
# mostly ~=1's and ~=0's
# the heat map does not show anything!
"""


def plot_cluster_heatmap_enhanced(probabilities):
    """
    Plot a heatmap of document-cluster probabilities with enhancements.

    Args:
        probabilities (np.ndarray): Probabilities of documents belonging to each cluster.
    """
    plt.figure(figsize=(12, 8))

    # Apply a log transformation (adding a small value to avoid log(0))
    log_probabilities = np.log(probabilities + 1e-10)

    # Use a diverging color palette
    sns.heatmap(log_probabilities, cmap='coolwarm', center=0)
    plt.title("Enhanced Heatmap of Document-Cluster Probabilities")
    plt.ylabel("Document Index")
    plt.xlabel("Cluster Index")
    plt.show()

# Call the enhanced plotting function
# plot_cluster_heatmap_enhanced(probabilities)


def plot_cluster_heatmap_filtered(probabilities, lower_bound=0.05, upper_bound=0.95):
    """
    Plot a heatmap of document-cluster probabilities, excluding values close to 0 or 1.

    Args:
        probabilities (np.ndarray): Probabilities of documents belonging to each cluster.
        lower_bound (float): Lower bound for filtering probabilities.
        upper_bound (float): Upper bound for filtering probabilities.
    """
    # Create a mask for values close to 0 or 1
    mask = np.logical_or(probabilities <= lower_bound,
                         probabilities >= upper_bound)

    plt.figure(figsize=(12, 8))
    sns.heatmap(probabilities, mask=mask, cmap='viridis')
    plt.title("Filtered Heatmap of Document-Cluster Probabilities")
    plt.ylabel("Document Index")
    plt.xlabel("Cluster Index")
    plt.show()
    a = 1


# Call the function with the probabilities
plot_cluster_heatmap_filtered(probabilities, 0.005, 0.995)

fitAndEvaluateGM(gm, X_lsa, labels=true_labels,
                 name="gm\nwith LSA on tf-idf vectors")
"""
clustering done in 2.07 ± 0.27 s 
Homogeneity: 0.628 ± 0.021
Completeness: 0.625 ± 0.018
V-measure: 0.626 ± 0.020
Adjusted Rand-Index: 0.655 ± 0.026
Normalized Mutual Info: 0.626 ± 0.020
Silhouette Coefficient: 0.043 ± 0.001
"""

printGMClusterTerms(lsa, gm, vectorizer, numClusters)


"""
Cluster 0: query data xml queries database problem algorithms time study based
Cluster 1: image images learning deep method visual training data using methods
Cluster 2: security attacks web privacy data paper based system secure users
Cluster 3: wireless networks network sensor nodes energy routing performance power protocol
"""
