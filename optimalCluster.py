from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import functools
from scipy.optimize import minimize
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import time
from typing import Callable
from itertools import product


def time_it(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with time measurement.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Time taken by {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


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


@time_it
def find_optimal_clusters_kmeans(X: np.ndarray, max_clusters: int,
                                 actualOptimalClusters: int = None,
                                 plot: bool = True,
                                 min_clusters: int = 2) -> Tuple[int, List[float], str]:
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

    rangeClusters = range(min_clusters, max_clusters + 1)

    with ProcessPoolExecutor() as executor:
        # Create a partial function with X already passed
        partial_fit_kmeans = functools.partial(fit_kmeans, X)

        # Map the partial function to the numbers and execute in parallel
        wcss = list(executor.map(partial_fit_kmeans, rangeClusters))

    # Automated Elbow Detection
    optimalClusters = find_elbow_point(wcss) + min_clusters

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
        # plt.show()

    return optimalClusters, wcss, filename


def evaluate_combinations(args: Tuple):
    return evaluate_gmm_cluster(*args)


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


def find_optimal_clusters_silhouette(data: np.ndarray, max_clusters=20,
                                     actualOptimalClusters: int = None, n_runs=5,
                                     min_clusters: int = 2):
    """
    Determine the optimal number of clusters using averaged performance metrics over multiple runs.
    """
    avg_silhouette_scores = []
    avg_aic_scores = []
    avg_bic_scores = []

    with ProcessPoolExecutor() as executor:
        for n_clusters in range(min_clusters, max_clusters + 1):
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
    plt.plot(range(min_clusters, max_clusters + 1),
             avg_silhouette_scores, marker='o')
    if actualOptimalClusters is not None:
        plt.axvline(x=actualOptimalClusters, color='blue',
                    linestyle='--',
                    label=f'Optimal Clusters: {actualOptimalClusters}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Average Silhouette Score vs. Number of Clusters')

    plt.subplot(1, 3, 2)
    plt.plot(range(min_clusters, max_clusters + 1), avg_aic_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average AIC Score')
    plt.title('Average AIC vs. Number of Clusters')

    plt.subplot(1, 3, 3)
    plt.plot(range(min_clusters, max_clusters + 1), avg_bic_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average BIC Score')
    plt.title('Average BIC vs. Number of Clusters')

    plt.tight_layout()

    # Saving the plot with a date-time stamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"optimalClusters/Silhouette_{timestamp}.png"
    plt.savefig(filename)
    # plt.show()

    # Choosing the optimal number of clusters based on the highest average silhouette score
    optimal_clusters = range(
        min_clusters, max_clusters + 1)[np.argmax(avg_silhouette_scores)]
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
                                        actualOptimalClusters: int = None,
                                        plot: bool = True,
                                        min_clusters: int = 2) -> Tuple[int, List[float], str]:
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

    rangeClusters = range(min_clusters, max_clusters + 1)

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
        optimalClusters = np.argmax(second_derivative) + min_clusters

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
        # plt.show()

    return optimalClusters, logLikelihoods, filename


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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
        ranked_clusters = [
            cluster for cluster in sorted_cluster_indices if prob[cluster] >= threshold]

        cluster_ranks.append(ranked_clusters)

    return cluster_ranks


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
    plt.savefig(f"plots/2D_Document_Clusters_{timestamp}.png")
    plt.show()


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
    plt.savefig(f"plots/3D_Document_Clusters_{timestamp}.png")
    plt.show()
