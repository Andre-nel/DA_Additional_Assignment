from bFinalUtils import find_optimal_clusters_kmeans, find_optimal_clusters_silhouette, optimal_clusters_gmm_modified_elbow
from time import time
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = set(stopwords.words('english'))

if __name__ == '__main__':
    # Read the CSV file
    data = pd.read_csv("D:/Andre/masters/DataAnalysis/additionalAssignment/arxiv2017.csv",
                       delimiter=';',
                       nrows=3000)
    # Add the 'Title' column to the 'Abstract' column and store it in a new column 'combined_text'
    data['combined_text'] = data['Title'] + ' ' + data['Abstract']

    # what categories are available? and how many?
    df = data.copy()
    categories = df['Subject_area'].unique()
    print(len(categories))
    print(categories)

    # List of selected subjects
    selected_subjects = ['LO', 'LG']
    columnToClean = 'combined_text'
    cleanedColumnName = 'cleaned'
    numClusters = len(selected_subjects)

    """
    # Preprocess data
    filtered_data = preprocess_data(columnToClean=columnToClean,
                                    cleanedColumnName=cleanedColumnName,
                                    selected_subjects=selected_subjects,
                                    numClusters=numClusters)
    """

    # filtered_data = pd.read_csv(
    #     'D:/Andre/masters/DataAnalysis/additionalAssignment/partB/filtered_data/2_clusters_2024-01-02_22-06-34.csv')

    filtered_data = pd.read_csv(
        'D:/Andre/masters/DataAnalysis/additionalAssignment/partB/filtered_data/10_clusters_2024-01-02_20-56-19.csv')

    print(filtered_data.shape)

    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(filtered_data['Subject_area'])

    minOccurrences = int(filtered_data.shape[0]*0.005)
    print(minOccurrences)

    # initialize the vectorizer
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, min_df=minOccurrences, max_df=0.85)

    X_tfidf = vectorizer.fit_transform(filtered_data[cleanedColumnName])

    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    t0 = time()
    X_lsa = lsa.fit_transform(X_tfidf)

    numClusters = 10
    maxClusters = numClusters*2
    print(maxClusters)

    # print(optimal_clusters_gmm_modified_elbow(X_lsa, maxClusters, numClusters))

    # print(find_optimal_clusters_silhouette(X_lsa, maxClusters, numClusters))

    print(find_optimal_clusters_kmeans(X_lsa, maxClusters, numClusters))

    a = 1
