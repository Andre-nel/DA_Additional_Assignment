# %%
# Quantifying the quality of clustering results
# =============================================
#
# In this section we define a function to score different clustering pipelines
# using several metrics.
#
# Clustering algorithms are fundamentally unsupervised learning methods.
# However, since we happen to have class labels for this specific dataset, it is
# possible to use evaluation metrics that leverage this "supervised" ground
# truth information to quantify the quality of the resulting clusters. Examples
# of such metrics are the following:
#
# - homogeneity, which quantifies how much clusters contain only members of a
#   single class;
#
# - completeness, which quantifies how much members of a given class are
#   assigned to the same clusters;
#
# - V-measure, the harmonic mean of completeness and homogeneity;
#
# - Rand-Index, which measures how frequently pairs of data points are grouped
#   consistently according to the result of the clustering algorithm and the
#   ground truth class assignment;
#
# - Adjusted Rand-Index, a chance-adjusted Rand-Index such that random cluster
#   assignment have an ARI of 0.0 in expectation.
#
# If the ground truth labels are not known, evaluation can only be performed
# using the model results itself. In that case, the Silhouette Coefficient comes
# in handy.
#
# For more reference, see :ref:`clustering_evaluation`.

from collections import defaultdict
from time import time
import numpy as np

from sklearn import metrics


def fit_and_evaluate_km(km, X, labels, name=None, n_runs=5):
    evaluations = []
    evaluations_std = []
    
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Normalized Mutual Info"].append(
            metrics.normalized_mutual_info_score(labels, km.labels_))
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)

    return evaluations, evaluations_std


from sklearn.mixture import GaussianMixture
from sklearn import metrics
from collections import defaultdict
from time import time
import numpy as np

def fitAndEvaluateGM(gm: GaussianMixture, x: np.ndarray, labels: np.ndarray, name: str = None, nRuns: int = 5) -> tuple[list[dict], list[dict]]:
    """
    Fit the Gaussian Mixture model and evaluate its performance.

    Args:
        gm (GaussianMixture): The Gaussian Mixture model to be fitted and evaluated.
        x (np.ndarray): The data to fit the model to.
        labels (np.ndarray): The true labels for the data.
        name (str, optional): The name of the estimator. Defaults to the class name of `gm` if None.
        nRuns (int, optional): The number of times to run the fitting for different seeds. Defaults to 5.

    Raises:
        TypeError: If `x` or `labels` is not a numpy ndarray.

    Returns:
        tuple[list[dict], list[dict]]: Two lists of dictionaries containing evaluation metrics and their standard deviations.

    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Expected x to be a numpy ndarray")
    if not isinstance(labels, np.ndarray):
        raise TypeError("Expected labels to be a numpy ndarray")

    evaluations = []
    evaluationsStd = []
    
    name = gm.__class__.__name__ if name is None else name

    trainTimes = []
    scores = defaultdict(list)
    for seed in range(nRuns):
        gm.set_params(random_state=seed)
        t0 = time()
        gm.fit(x)
        trainTimes.append(time() - t0)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, gm.predict(x)))
        scores["Completeness"].append(metrics.completeness_score(labels, gm.predict(x)))
        scores["V-measure"].append(metrics.v_measure_score(labels, gm.predict(x)))
        scores["Adjusted Rand-Index"].append(metrics.adjusted_rand_score(labels, gm.predict(x)))
        scores["Normalized Mutual Info"].append(metrics.normalized_mutual_info_score(labels, gm.predict(x)))
        scores["Silhouette Coefficient"].append(metrics.silhouette_score(x, gm.predict(x), sample_size=2000))
    trainTimes = np.asarray(trainTimes)

    print(f"clustering done in {trainTimes.mean():.2f} ± {trainTimes.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": trainTimes.mean(),
    }
    evaluationStd = {
        "estimator": name,
        "train_time": trainTimes.std(),
    }
    for scoreName, scoreValues in scores.items():
        meanScore, stdScore = np.mean(scoreValues), np.std(scoreValues)
        print(f"{scoreName}: {meanScore:.3f} ± {stdScore:.3f}")
        evaluation[scoreName] = meanScore
        evaluationStd[scoreName] = stdScore
    evaluations.append(evaluation)
    evaluationsStd.append(evaluationStd)

    return evaluations, evaluationsStd


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture
import numpy as np

def printGMClusterTerms(lsa: TruncatedSVD, gm: GaussianMixture, vectorizer: CountVectorizer, numClusters: int) -> None:
    """
    Print the top terms in each cluster.

    Args:
        lsa (TruncatedSVD): The fitted LSA (Latent Semantic Analysis) model.
        gm (GaussianMixture): The fitted Gaussian Mixture model.
        vectorizer (CountVectorizer): The vectorizer used to fit the LSA model.
        numClusters (int): The number of clusters.

    Raises:
        TypeError: If any of the inputs are not of the expected type.
    """
    if not isinstance(lsa[0], TruncatedSVD):
        raise TypeError("Expected lsa to be an instance of TruncatedSVD")
    if not isinstance(gm, GaussianMixture):
        raise TypeError("Expected gm to be an instance of GaussianMixture")
    if not isinstance(vectorizer, CountVectorizer):
        raise TypeError("Expected vectorizer to be an instance of CountVectorizer")
    if not isinstance(numClusters, int):
        raise TypeError("Expected numClusters to be an integer")

    originalSpaceCentroids = lsa[0].inverse_transform(gm.means_)
    orderCentroids = originalSpaceCentroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(numClusters):
        print(f"Cluster {i}: ", end="")
        for ind in orderCentroids[i, :10]:
            print(f"{terms[ind]} ", end="")
        print()


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

def findOptimalClusters(data, max_clusters, nRuns=5):
    """
    Determine the optimal number of clusters using averaged performance metrics over multiple runs.

    Args:
        data (np.array): The data to be clustered.
        max_clusters (int): The maximum number of clusters to try.
        nRuns (int): The number of runs for each number of clusters to average the metrics.

    Returns:
        int: Optimal number of clusters.
    """
    avg_silhouette_scores = []
    avg_aic_scores = []
    avg_bic_scores = []

    for n_clusters in range(2, max_clusters + 1):
        silhouette_scores = []
        aic_scores = []
        bic_scores = []

        for run in range(nRuns):
            gm = GaussianMixture(n_components=n_clusters, max_iter=100, n_init=30, random_state=run)
            labels = gm.fit_predict(data)

            silhouette_scores.append(silhouette_score(data, labels))
            aic_scores.append(gm.aic(data))
            bic_scores.append(gm.bic(data))

        # Averaging the scores over the runs
        avg_silhouette_scores.append(np.mean(silhouette_scores))
        avg_aic_scores.append(np.mean(aic_scores))
        avg_bic_scores.append(np.mean(bic_scores))

        print(f"Clusters: {n_clusters}, Avg Silhouette: {np.mean(silhouette_scores):.3f}, Avg AIC: {np.mean(aic_scores):.3f}, Avg BIC: {np.mean(bic_scores):.3f}")

    # Plotting the results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(range(2, max_clusters + 1), avg_silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Average Silhouette Score vs. Number of Clusters')

    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_clusters + 1), avg_aic_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average AIC Score')
    plt.title('Average AIC vs. Number of Clusters')

    plt.subplot(1, 3, 3)
    plt.plot(range(2, max_clusters + 1), avg_bic_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average BIC Score')
    plt.title('Average BIC vs. Number of Clusters')

    plt.tight_layout()

    # Saving the plot with a date-time stamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"cluster_evaluation_{timestamp}.png"
    plt.savefig(filename)
    plt.show()
    # plt.close()

    # Choosing the optimal number of clusters based on the highest average silhouette score
    optimal_clusters = range(2, max_clusters + 1)[np.argmax(avg_silhouette_scores)]
    print(f"Optimal number of clusters based on average Silhouette Score: {optimal_clusters}")

    return optimal_clusters

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def findOptimalClustersGMM(X, max_clusters):
    """
    Determine the optimal number of clusters using the elbow method for Gaussian Mixture Models.

    Args:
        X (np.array): The data to be clustered.
        max_clusters (int): The maximum number of clusters to try.

    Returns:
        int: Optimal number of clusters.
    """
    log_likelihoods = []
    range_clusters = range(1, max_clusters + 1)

    for n_clusters in range_clusters:
        gmm = GaussianMixture(n_components=n_clusters, max_iter=100, n_init=30)
        gmm.fit(X)
        log_likelihoods.append(gmm.score(X))

    # Plotting the log-likelihoods
    plt.figure(figsize=(8, 4))
    plt.plot(range_clusters, log_likelihoods, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Log-Likelihood')
    plt.title('GMM Log-Likelihood vs. Number of Clusters')
    # Saving the plot with a date-time stamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"cluster_evaluation_logLikelyhood_{timestamp}.png"
    plt.savefig(filename)
    plt.show()

    # Finding the 'elbow' in the log-likelihood curve can be subjective. 
    # You might need to choose the number of clusters based on where you see a diminishing return in log-likelihood increase.
    # There's no automatic way to find the elbow, as it's a visual and heuristic method.
    # Return the chosen optimal clusters based on your observation.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime

def findOptimalClustersKMeans(X: np.ndarray, maxClusters: int) -> int:
    """
    Determine the optimal number of clusters for K-Means using the elbow method.

    This function computes K-Means clustering for a range of cluster values and plots the within-cluster sum of squares (WCSS) for each. The 'elbow' point in this plot typically represents the optimal number of clusters.

    Args:
        X (np.ndarray): The data to be clustered.
        maxClusters (int): The maximum number of clusters to try.

    Returns:
        int: Optimal number of clusters, identified visually from the plot.

    Raises:
        ValueError: If maxClusters is less than 1 or X is not a 2D array.

    """

    if maxClusters < 1:
        raise ValueError("maxClusters must be at least 1.")
    # if not isinstance(X, np.ndarray) or len(X.shape) != 2:
    #     raise ValueError("X must be a 2D numpy array.")

    wcss = []
    rangeClusters = range(1, maxClusters + 1)

    for nClusters in rangeClusters:
        kmeans = KMeans(n_clusters=nClusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(rangeClusters, wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('K-Means WCSS vs. Number of Clusters')
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"KMeans_elbow_method_{timestamp}.png"
    plt.savefig(filename)
    plt.show()

    # The optimal number of clusters is determined visually
    return -1  # Placeholder, update after visual examination of the plot

# Example usage
# optimalClusters = findOptimalClustersKMeans(data, maxClusters=10)
# print(f"Optimal Number of Clusters: {optimalClusters}")
