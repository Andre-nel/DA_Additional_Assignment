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

X = vectorizer.fit_transform(filtered_data[cleanedColumnName])

# Initialize KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
# Fit the model
kmeans.fit(X)
# Store cluster labels in a variable
clusters = kmeans.labels_

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from sklearn.decomposition import PCA

# Initialize PCA with 3 components
pca = PCA(n_components=3, random_state=42)
pca_vecs = pca.fit_transform(X.toarray())
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]
x2 = pca_vecs[:, 2]

filtered_data['cluster'] = clusters
filtered_data['x0'] = x0
filtered_data['x1'] = x1
filtered_data['x2'] = x2

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis

# Set title and axis labels
ax.set_title("3D PCA Plot", fontsize=18)
ax.set_xlabel("X0", fontsize=16)
ax.set_ylabel("X1", fontsize=16)
ax.set_zlabel("X2", fontsize=16)

# Create a 3D scatter plot with cluster coloring
scatter = ax.scatter(
    filtered_data['x0'],
    filtered_data['x1'],
    filtered_data['x2'],
    c=filtered_data['cluster']
)

# Add a color bar for cluster mapping
plt.colorbar(scatter, label='Cluster')
plt.show()

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari_score = adjusted_rand_score(true_labels, filtered_data['cluster'])
nmi_score = normalized_mutual_info_score(true_labels, filtered_data['cluster'])

print(f"Adjusted Rand Index (ARI): {ari_score}")
print(f"Normalized Mutual Information (NMI): {nmi_score}")

a = 1