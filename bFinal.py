# from bFinalUtils import (find_optimal_clusters_kmeans, find_optimal_clusters_silhouette, dropColumnsNotInList,
#                          optimal_clusters_gmm_modified_elbow, preprocessData, preprocess_data)

from preprocess import preprocessData as ppd
from optimalCluster import (find_optimal_clusters_kmeans, optimal_clusters_gmm_modified_elbow,
                            find_optimal_clusters_silhouette, plot_clusters, plot_clusters_3d,
                            assign_cluster_ranks)

from time import time
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = set(stopwords.words('english'))

if __name__ == '__main__':
    # Read the CSV file
    # data = pd.read_csv("D:/Andre/masters/DataAnalysis/additionalAssignment/arxiv2017.csv",
    #                    delimiter=';',
    #                    )
    # # Add the 'Title' column to the 'Abstract' column and store it in a new column 'combined_text'
    # data['cleaned'] = data['Title'] + ' ' + data['Abstract']

    # # Optimize DataFrame memory usage
    # data = dropColumnsNotInList(
    #     data, ['ID', 'Subject_area', 'cleaned'])

    # # what categories are available? and how many?
    # categories = data['Subject_area'].unique()
    # print(len(categories))
    # print(categories)

    # # List of selected subjects
    # selected_subjects = []  # select all
    # columnToClean = 'cleaned'
    cleanedColumnName = 'cleaned'
    # if selected_subjects:
    #     numClusters = len(selected_subjects)
    # else:
    #     numClusters = len(categories)

    # print(data.shape)
    # start_time = time()
    # # Preprocess data
    # filtered_data = ppd(data=data, columnToClean=columnToClean,
    #                     cleanedColumnName=cleanedColumnName,
    #                     numClusters=numClusters)
    # end_time = time()
    # print(f"Time taken by ppd: {end_time - start_time} seconds") Time taken by ppd: 90.35864663124084 seconds

    # filtered_data = pd.read_csv(
    #     'D:/Andre/masters/DataAnalysis/additionalAssignment/partB/filtered_data/2_clusters_2024-01-02_22-06-34.csv')

    # filtered_data = pd.read_csv(
    #     'D:/Andre/masters/DataAnalysis/additionalAssignment/partB/filtered_data/10_clusters_2024-01-02_20-56-19.csv')

    # filtered_data = pd.read_csv(
    #     'D:/Andre/masters/DataAnalysis/additionalAssignment/DA_Additional_Assignment/filtered_data/40_131565_ppd_clusters_2024-01-05_21-26-59.csv')

    # print(filtered_data.shape)

    # label_encoder = LabelEncoder()
    # true_labels = label_encoder.fit_transform(filtered_data['Subject_area'])

    # minOccurrences = int(filtered_data.shape[0]*0.005)
    # print(minOccurrences)

    # # initialize the vectorizer
    # vectorizer = TfidfVectorizer(
    #     sublinear_tf=True, min_df=minOccurrences, max_df=0.80)

    # # Replace np.nan values with an empty string
    # filtered_data[cleanedColumnName].fillna('', inplace=True)
    # X_tfidf = vectorizer.fit_transform(filtered_data[cleanedColumnName])

    # lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    # t0 = time()
    # X_lsa = lsa.fit_transform(X_tfidf)

    # save the ndarray to a format that can be loaded fast.
    # Save the ndarray to a file
    # _, cols = X_tfidf.shape
    # np.save(f'X_lsa/X_lsa_{cols}.npy', X_lsa)

    # Later, to load the array back into memory
    X_lsa_loaded: np.ndarray = np.load('X_lsa/X_lsa_2532.npy')

    numClusters = 40
    minClusters = 5
    maxClusters = 40

    # print(find_optimal_clusters_kmeans(X_lsa_loaded.copy(),
    #       maxClusters, numClusters, min_clusters=minClusters))

    # print(optimal_clusters_gmm_modified_elbow(
    #     X_lsa_loaded.copy(), maxClusters, numClusters, min_clusters=minClusters))

    # print(find_optimal_clusters_silhouette(
    #     X_lsa_loaded.copy(), maxClusters, numClusters, min_clusters=minClusters))

    optNumClusters = 15

    # todo find the optimal number of clusters
    optNumClusters = 3
    from sklearn.mixture import GaussianMixture

    gm = GaussianMixture(
        n_components=optNumClusters,
        max_iter=100,
        n_init=30,
    )

    gm.fit(X_lsa_loaded)

    # Assuming 'gm' is your fitted GaussianMixture model and 'X' is your data
    probabilities = gm.predict_proba(X_lsa_loaded)

    # assign clusters, primary, secondary, tertiary etc
    cluster_assignments = assign_cluster_ranks(probabilities)

    # Example: Print cluster assignments for the first few documents
    for i, ranks in enumerate(cluster_assignments):
        if len(ranks) > 1:
            print(f"Document {i}: Cluster Ranks: {ranks}")

    """# Documents 512 and 642 were assigned to
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
    # X_reduced = tsne.fit_transform(X_lsa)"""

    from sklearn.decomposition import PCA
    # Initialize PCA
    pca_2 = PCA(n_components=2, random_state=0)
    # Fit and transform the data
    X_reduced = pca_2.fit_transform(X_lsa_loaded)

    plot_clusters(X_reduced, cluster_assignments)

    pca_3 = PCA(n_components=3, random_state=0)

    # Fit and transform the data
    X_reduced_3d = pca_3.fit_transform(X_lsa_loaded)

    # Example usage
    plot_clusters_3d(X_reduced_3d, cluster_assignments)
