
# import required sklearn libs
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# import other required libs
import pandas as pd
import numpy as np

# string manipulation libs
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This function cleans the input text by
    - removing links
    - removing special chars
    - removing numbers
    - removing stopwords
    - transforming in lower case
    - removing excessive whitespaces
    Arguments:
        text (str): text to clean
        remove_stopwords (bool): remove stopwords or not
    Returns:
        str: cleaned text
    """
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove numbers and special chars
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. creates tokens
        tokens = nltk.word_tokenize(text)
        # 2. checks if token is a stopword and removes it
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. joins all tokens again
        text = " ".join(tokens)
    # returns cleaned text
    text = text.lower().strip()
    return text
  
def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups tf idf vector per cluster
    terms = vectorizer.get_feature_names_out() # access to tf idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
   

categories = [
 'rec.sport.baseball',
 'alt.atheism',
 'soc.religion.christian',
]
# dataset = fetch_20newsgroups(subset='train',
#                              categories=categories,
#                              shuffle=True,
#                              remove=('headers', 'footers', 'quotes'))

# df = pd.DataFrame(dataset.data, columns=["corpus"])
# df['cleaned'] = df['corpus'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

# # Save the filtered DataFrame to a CSV file
# from datetime import datetime
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# filename = f"df_corpus_{timestamp}.csv"
# df.to_csv(filename, index=False)

df = pd.read_csv('C:/Users/Jakne/Desktop/Andre/df_corpus_2023-12-30_15-40-57.csv')

df['cleaned'].fillna('', inplace=True)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.80)

X_tfidf = vectorizer.fit_transform(df['cleaned'])

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time

lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
X_lsa = lsa.fit_transform(X_tfidf)
explained_variance = lsa[0].explained_variance_ratio_.sum()

# todo find the optimal number of clusters
numClusters = 3
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(
    n_components=numClusters,
    max_iter=100,
    n_init=30,
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.mixture import GaussianMixture

gm.fit(X_lsa)

# Assuming 'gm' is your fitted GaussianMixture model and 'X' is your data
probabilities = gm.predict_proba(X_lsa)


# todo assign clusters, primary, secondary, tertiary etc.
# if the probability for a category is greater than 0.005 it gets
# assigned the the clusters in the corresponding rank that it falls in (primary, secondary, ...)
def assign_cluster_ranks(probabilities, threshold=0.005):
    """
    Assigns cluster ranks (primary, secondary, etc.) to each document based on probabilities.

    Args:
        probabilities (np.ndarray): Probabilities of documents belonging to each cluster.
        threshold (float): Threshold for including a cluster in the ranking.

    Returns:
        List[List[int]]: A list of lists containing cluster ranks for each document.
    """
    cluster_ranks = []

    for prob in probabilities:
        # Sort the clusters by probability in descending order
        sorted_cluster_indices = np.argsort(prob)[::-1]

        # Filter out clusters with probabilities below the threshold
        ranked_clusters = [cluster for cluster in sorted_cluster_indices if prob[cluster] >= threshold]

        cluster_ranks.append(ranked_clusters)

    return cluster_ranks

# Call the function with the probabilities
cluster_assignments = assign_cluster_ranks(probabilities)

# Example: Print cluster assignments for the first few documents
for i, ranks in enumerate(cluster_assignments):
    if len(ranks) > 1:
        print(f"Document {i}: Cluster Ranks: {ranks}")

# Documents 512 and 642 were assigned to
# both the clusters 0 and 2
# cluster_assignments[642] = [2, 0]
# we are to reduce the dimensionality of X_lsa to 2 dimensions and plot the clusters,
# colour coding the dot's according to clusters, the articles that are assigned
# secondary and tertiary clusters are to be
# easily spotted with their distinct colours
# and shapes. Hopefully the documents that have been assigned secondary clusters lie on the boundaries or where the clusters overlap in the plot.
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# # Reduce dimensions
# tsne = TSNE(n_components=2, random_state=0)
# X_reduced = tsne.fit_transform(X_lsa)

from sklearn.decomposition import PCA
# Initialize PCA
pca = PCA(n_components=2, random_state=0)
# Fit and transform the data
X_reduced = pca.fit_transform(X_lsa)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(X_reduced, cluster_assignments, primary_color_map='viridis'):
    """
    Plots the clusters with distinct colors and shapes for documents having secondary/tertiary clusters.
    Documents with secondary clusters are represented with bigger, red markers.

    Args:
        X_reduced (np.ndarray): 2D reduced representation of the data.
        cluster_assignments (List[List[int]]): List of cluster rankings for each document.
        primary_color_map (str): Colormap for primary clusters.
    """
    plt.figure(figsize=(12, 8))

    # Flatten the cluster_assignments list to find unique clusters
    flat_clusters = [cluster for doc in cluster_assignments for cluster in doc]
    unique_clusters = np.unique(flat_clusters)

    # Create a colormap
    colors = plt.cm.get_cmap(primary_color_map, len(unique_clusters))

    # Plot each document
    for i, (x, y) in enumerate(X_reduced):
        primary_cluster = cluster_assignments[i][0] if cluster_assignments[i] else -1
        color = colors(unique_clusters.tolist().index(primary_cluster))

        # Check if there are secondary clusters and adjust the marker and color accordingly
        if len(cluster_assignments[i]) > 1:
            marker = '*'  # Different marker for secondary clusters
            color = 'red'  # Red color for secondary clusters
            size = 100  # Bigger size for secondary clusters
        else:
            marker = 'o'
            size = 50  # Default size for primary clusters

        plt.scatter(x, y, color=color, marker=marker, s=size, alpha=0.7)

    plt.title("2D Visualization of Document Clusters")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.savefig(f"2D_Document_Clusters_{timestamp}.png")
    plt.show()
plot_clusters(X_reduced, cluster_assignments)


# 0 and 1
# 481, 512, 737, 1490

from sklearn.decomposition import PCA

# Assuming X_lsa is your high-dimensional data
# Choose 3 components for 3D visualization
pca = PCA(n_components=3, random_state=0)

# Fit and transform the data
X_reduced_3d = pca.fit_transform(X_lsa)

from mpl_toolkits.mplot3d import Axes3D

def plot_clusters_3d(X_reduced_3d, cluster_assignments, primary_color_map='viridis'):
    """
    Plots the clusters in 3D with distinct colors and shapes for documents having secondary/tertiary clusters.
    Documents with secondary clusters are represented with bigger, red markers.

    Args:
        X_reduced_3d (np.ndarray): 3D reduced representation of the data.
        cluster_assignments (List[List[int]]): List of cluster rankings for each document.
        primary_color_map (str): Colormap for primary clusters.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Flatten the cluster_assignments list to find unique clusters
    flat_clusters = [cluster for doc in cluster_assignments for cluster in doc]
    unique_clusters = np.unique(flat_clusters)

    # Create a colormap
    colors = plt.cm.get_cmap(primary_color_map, len(unique_clusters))

    # Plot each document
    for i, (x, y, z) in enumerate(X_reduced_3d):
        primary_cluster = cluster_assignments[i][0] if cluster_assignments[i] else -1
        color = colors(unique_clusters.tolist().index(primary_cluster))

        # Check if there are secondary clusters and adjust the marker and color accordingly
        if len(cluster_assignments[i]) > 1:
            marker = '*'  # Different marker for secondary clusters
            color = 'red'  # Red color for secondary clusters
            size = 100  # Bigger size for secondary clusters
        else:
            marker = 'o'
            size = 50  # Default size for primary clusters

        ax.scatter(x, y, z, color=color, marker=marker, s=size, alpha=0.7)

    ax.set_title("3D Visualization of Document Clusters")
    ax.set_xlabel("PCA Feature 1")
    ax.set_ylabel("PCA Feature 2")
    ax.set_zlabel("PCA Feature 3")
    plt.savefig(f"3D_Document_Clusters_{timestamp}.png")
    plt.show()

# Example usage
plot_clusters_3d(X_reduced_3d, cluster_assignments)




# initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
clusters = kmeans.labels_

# initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)
# pass X to the pca
pca_vecs = pca.fit_transform(X.toarray())
# save the two dimensions in x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

# assign clusters and PCA vectors to columns in the original dataframe
df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1

cluster_map = {0: "sport", 1: "technology", 2: "religion"} # mapping found through get_top_keywords
df['cluster'] = df['cluster'].map(cluster_map)

# set image size
plt.figure(figsize=(12, 7))
# set title
plt.title("Raggruppamento TF-IDF + KMeans 20newsgroup", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
#  create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="viridis")
plt.show()         