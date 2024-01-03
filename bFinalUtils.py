from itertools import product
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import functools

from datetime import datetime

nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = set(stopwords.words('english'))


def preprocess_data(data, columnToClean: str, cleanedColumnName: str,
                    selected_subjects: list, numClusters) -> pd.DataFrame:
    """
    Filter it by selected subjects, and preprocess the text.

    Args:
        selected_subjects (list): List of subject areas to filter the data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Filter data by selected subjects
    filtered_data: pd.DataFrame = data[data['Subject_area'].isin(
        selected_subjects)]

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
    filename = f"./filtered_data/{numClusters}_clusters_{timestamp}.csv"

    filtered_data.to_csv(filename, index=False)

    return filtered_data


def evaluate_gmm_cluster(data, n_clusters, run):
    """
    Evaluate a single clustering run using Gaussian Mixture Model.

    Args:
        data (np.array): The data to be clustered.
        n_clusters (int): Number of clusters for Gaussian Mixture.
        run (int): Seed for random state in Gaussian Mixture.

    Returns:
        tuple: silhouette score, AIC score, BIC score for the run.
    """
    gm = GaussianMixture(n_components=n_clusters,
                         max_iter=100, n_init=30, random_state=run)
    labels = gm.fit_predict(data)
    return silhouette_score(data, labels), gm.aic(data), gm.bic(data)


def evaluate_combinations(args: Tuple):
    return evaluate_gmm_cluster(*args)


def find_optimal_clusters_silhouette(data: np.ndarray, max_clusters=20,
                                     actualOptimalClusters: int = None, n_runs=5):
    """
    Determine the optimal number of clusters using averaged performance metrics over multiple runs.
    """
    avg_silhouette_scores = []
    avg_aic_scores = []
    avg_bic_scores = []

    with ProcessPoolExecutor() as executor:
        for n_clusters in range(2, max_clusters + 1):
            print(f"Processing {n_clusters} clusters...")

            # Create all combinations of n_clusters and runs
            combinations = list(
                product([data.copy()], [n_clusters], range(n_runs)))

            # Use map to apply evaluate_gmm_cluster to all combinations
            results = executor.map(evaluate_combinations, combinations)

            silhouette_scores, aic_scores, bic_scores = zip(*results)

            # Averaging the scores over the runs
            avg_silhouette_scores.append(np.mean(silhouette_scores))
            avg_aic_scores.append(np.mean(aic_scores))
            avg_bic_scores.append(np.mean(bic_scores))

            print(
                f"Clusters: {n_clusters}, Avg Silhouette: {np.mean(silhouette_scores):.3f}, "
                f"Avg AIC: {np.mean(aic_scores):.3f}, Avg BIC: {np.mean(bic_scores):.3f}")

    # Plotting the results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(range(2, max_clusters + 1), avg_silhouette_scores, marker='o')
    if actualOptimalClusters is not None:
        plt.axvline(x=actualOptimalClusters, color='blue',
                    linestyle='--',
                    label=f'Optimal Clusters: {actualOptimalClusters}')
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
    filename = f"optimalClusters/Silhouette_{timestamp}.png"
    plt.savefig(filename)
    plt.show()

    # Choosing the optimal number of clusters based on the highest average silhouette score
    optimal_clusters = range(
        2, max_clusters + 1)[np.argmax(avg_silhouette_scores)]
    print(
        f"Optimal number of clusters based on average Silhouette Score: {optimal_clusters}")

    return optimal_clusters


def fit_gmm(X: np.array, n_clusters: int) -> float:
    """
    Fit Gaussian Mixture Model and return the log-likelihood.

    Args:
        X (np.array): The data to be clustered.
        n_clusters (int): Number of clusters for Gaussian Mixture.

    Returns:
        float: Log-likelihood of the model.
    """
    gmm = GaussianMixture(n_components=n_clusters, max_iter=100, n_init=30)
    gmm.fit(X)
    return gmm.score(X)


def optimal_clusters_gmm_modified_elbow(X: np.array, max_clusters: int,
                                        actualOptimalClusters: int = None, plot: bool = True) -> Tuple[int, List[float], str]:
    """
    Determine the optimal number of clusters using a modified elbow method for Gaussian Mixture Models.

    Args:
        X (np.array): The data to be clustered.
        max_clusters (int): The maximum number of clusters to try.
        plot (bool): Whether to plot the log-likelihood graph. Defaults to True.

    Returns:
        Tuple[int, List[float], str]: Optimal number of clusters, log-likelihoods, and plot file name.
    """
    if max_clusters < 1:
        raise ValueError("maxClusters must be a positive integer")

    rangeClusters = range(1, max_clusters + 1)

    with ProcessPoolExecutor(max_workers=8) as executor:
        partial_fit_gmm = functools.partial(fit_gmm, X)

        # Using the helper function to unpack arguments
        logLikelihoods = list(executor.map(partial_fit_gmm, rangeClusters))

    # Automated Elbow Detection (simple approach)
    optimalClusters = max_clusters
    if len(logLikelihoods) > 2:
        # First derivative
        first_derivative = np.diff(logLikelihoods)

        # Second derivative
        second_derivative = np.diff(first_derivative)

        # Find the elbow point as the first point of maximum curvature
        optimalClusters = np.argmax(second_derivative) + 1

    filename = ""
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(rangeClusters, logLikelihoods, marker='o')
        plt.axvline(x=optimalClusters, color='red',
                    linestyle='--',
                    label=f'Optimal Clusters: {optimalClusters}')
        if actualOptimalClusters is not None:
            plt.axvline(x=actualOptimalClusters, color='blue',
                        linestyle='--',
                        label=f'Optimal Clusters: {actualOptimalClusters}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Log-Likelihood')
        plt.title('GMM Log-Likelihood vs. Number of Clusters')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"optimalClusters/logLikelihood_{timestamp}.png"
        plt.savefig(filename)
        plt.show()

    return optimalClusters, logLikelihoods, filename


# def find_elbow_point(wcss: List[float]) -> int:
#     """
#     Find the elbow point in the WCSS (within-cluster sum of squares) curve.

#     Args:
#         wcss (List[float]): List of WCSS values for different numbers of clusters.

#     Returns:
#         int: Optimal number of clusters based on the elbow method.
#     """
#     if len(wcss) <= 2:
#         return len(wcss)

#     # Smooth the WCSS values using a simple moving average
#     window_size = 3
#     smoothed_wcss = np.convolve(wcss, np.ones(
#         window_size) / window_size, mode='valid')

#     # Calculate the second derivative
#     second_derivative = np.diff(smoothed_wcss, n=2)

#     # Find the elbow point
#     elbow_point = np.argmax(second_derivative) + 1 + (window_size - 1) // 2

#     return elbow_point


def find_elbow_point(wcss: List[float]) -> int:
    """
    Find the elbow point using two-line segmentation.

    Args:
        wcss (List[float]): List of WCSS values for different numbers of clusters.

    Returns:
        int: Optimal number of clusters based on the elbow method.
    """

    # Define the error function for the piecewise linear fit
    def fit_error(split_point, wcss, n):
        line1 = np.polyfit(range(1, split_point + 1), wcss[:split_point], 1)
        line2 = np.polyfit(range(split_point, n + 1),
                           wcss[split_point - 1:], 1)

        error1 = sum(
            (np.polyval(line1, range(1, split_point + 1)) - wcss[:split_point]) ** 2)
        error2 = sum(
            (np.polyval(line2, range(split_point, n + 1)) - wcss[split_point - 1:]) ** 2)

        return error1 + error2

    n = len(wcss)
    split_point = minimize(lambda x: fit_error(
        int(x), wcss, n), x0=2, bounds=[(2, n)]).x
    return int(split_point)


def fit_kmeans(X: np.ndarray, n_clusters: int) -> float:
    """
    Fit KMeans and return the within-cluster sum of squares (WCSS).

    Args:
        X (np.ndarray): The data to be clustered.
        n_clusters (int): Number of clusters for KMeans.

    Returns:
        float: WCSS for the model.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                    max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    return kmeans.inertia_


def find_optimal_clusters_kmeans(X: np.ndarray, max_clusters: int,
                                 actualOptimalClusters: int = None, plot: bool = True) -> Tuple[int, List[float], str]:
    """
    Determine the optimal number of clusters for K-Means using a modified elbow method.

    Args:
        X (np.ndarray): The data to be clustered.
        max_clusters (int): The maximum number of clusters to try.
        plot (bool): Whether to plot the WCSS graph. Defaults to True.

    Returns:
        Tuple[int, List[float], str]: Optimal number of clusters, WCSS values, and plot file name.
    """
    if max_clusters < 1:
        raise ValueError("maxClusters must be at least 1.")

    rangeClusters = range(1, max_clusters + 1)

    with ProcessPoolExecutor() as executor:
        # Create a partial function with X already passed
        partial_fit_kmeans = functools.partial(fit_kmeans, X)

        # Map the partial function to the numbers and execute in parallel
        wcss = list(executor.map(partial_fit_kmeans, rangeClusters))

    # Automated Elbow Detection
    optimalClusters = find_elbow_point(wcss)

    filename = ""
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(rangeClusters, wcss, marker='o')
        plt.axvline(x=optimalClusters, color='red', linestyle='--',
                    label=f'Optimal Clusters: {optimalClusters}')
        if actualOptimalClusters is not None:
            plt.axvline(x=actualOptimalClusters, color='blue',
                        linestyle='--',
                        label=f'Optimal Clusters: {actualOptimalClusters}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('K-Means WCSS vs. Number of Clusters')
        plt.legend()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"optimalClusters/KMeans_elbow_method_{timestamp}.png"
        plt.savefig(filename)
        plt.show()

    return optimalClusters, wcss, filename
