from typing import List, Tuple
from multiprocessing import Pool
from itertools import product
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
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


def preprocessData(data: pd.DataFrame, columnToClean: str, cleanedColumnName: str,
                   selectedSubjects: List[str], numClusters: int) -> pd.DataFrame:
    """
    Preprocesses a DataFrame by filtering it by selected subjects and cleaning text in parallel.

    Args:
        data (pd.DataFrame): The DataFrame to preprocess.
        columnToClean (str): The name of the column containing text to clean.
        cleanedColumnName (str): The name of the column to store cleaned text.
        selectedSubjects (List[str]): Subjects to filter the DataFrame.
        numClusters (int): Number of clusters for naming the output file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Filter data by selected subjects
    data = data[data['Subject_area'].isin(
        selectedSubjects)] if selectedSubjects else data.copy()

    data = data.rename(
        columns={columnToClean: cleanedColumnName})

    # Optimize DataFrame memory usage
    data = dropColumnsNotInList(
        data, ['ID', 'Subject_area', cleanedColumnName])

    # Split dataframe for multiprocessing
    # Adjust number of partitions based on DataFrame size
    numPartitions = min(10, len(data))
    dfSplit = np.array_split(data, numPartitions)

    # Initialize multiprocessing pool
    try:
        pool = Pool()
        partialPreprocessAndClean = functools.partial(
            preprocessAndClean, cleanedColumnName)
        dfProcessed: pd.DataFrame = pd.concat(
            pool.map(partialPreprocessAndClean, dfSplit))
    finally:
        pool.close()
        pool.join()

    # Save the filtered DataFrame to a CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./filtered_data/{numClusters}_clusters_{timestamp}.csv"
    dfProcessed.to_csv(filename, index=False)

    return dfProcessed


def preprocessAndClean(cleanedColumnName: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a specific column in a chunk of DataFrame.

    Args:
        cleanedColumnName (str): Column name for cleaned text also the column to clean.
        df (pd.DataFrame): DataFrame chunk to process.

    Returns:
        pd.DataFrame: Processed DataFrame chunk.
    """
    df[cleanedColumnName] = df[cleanedColumnName].apply(
        lambda x: preprocessText(x))
    return df


def preprocessText(text: str, removeStopwords: bool = True) -> str:
    """
    Preprocesses text by removing unwanted characters and optionally stopwords.

    Args:
        text (str): The text to preprocess.
        removeStopwords (bool): Flag to indicate if stopwords should be removed.

    Returns:
        str: The preprocessed text.
    """
    text = re.sub(r"http\S+", "", text)  # Remove links
    if removeStopwords:
        tokens = nltk.word_tokenize(text)
        tokens = [w for w in tokens if not w.lower()
                  in stopwords.words("english")]
        text = " ".join(tokens)
    return text.lower().strip()


def dropColumnsNotInList(df: pd.DataFrame, columns_to_keep: List[str]) -> pd.DataFrame:
    """
    Drops columns from a DataFrame that are not in the specified list.

    Args:
        df (pd.DataFrame): The DataFrame from which to drop columns.
        columns_to_keep (List[str]): A list of column names to keep.

    Returns:
        pd.DataFrame: The DataFrame with only the specified columns retained.

    Raises:
        ValueError: If any of the columns in `columns_to_keep` do not exist in `df`.

    """
    # Check if all columns to keep are in the DataFrame
    for column in columns_to_keep:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Determine which columns to drop
    columns_to_drop = df.columns.difference(columns_to_keep)

    # Drop the unwanted columns
    df_dropped = df.drop(columns=columns_to_drop)

    return df_dropped

# def preprocess_data(data: pd.DataFrame, columnToClean: str, cleanedColumnName: str,
#                     selected_subjects: List[str], numClusters: int) -> pd.DataFrame:
#     """
#     Filter it by selected subjects, and preprocess the text in parallel using multiprocessing.

#     Args:
#         data (pd.DataFrame): Input DataFrame.
#         columnToClean (str): The name of the column to clean.
#         cleanedColumnName (str): The name of the column to store cleaned data.
#         selected_subjects (List[str]): List of subject areas to filter the data.
#         numClusters (int): Number of clusters for naming the output file.

#     Returns:
#         pd.DataFrame: Preprocessed DataFrame.
#     """
#     # Filter data by selected subjects
#     if selected_subjects:
#         filtered_data = data[data['Subject_area'].isin(selected_subjects)]
#     else:
#         filtered_data = data.copy()

#     # Split dataframe for multiprocessing
#     num_partitions = 10  # Number of partitions to split dataframe
#     df_split = np.array_split(filtered_data, num_partitions)

#     pool = Pool()
#     partial_preprocess_and_clean = functools.partial(preprocess_and_clean,
#                                                      cleanedColumnName,
#                                                      columnToClean)
#     df_processed = pd.concat(pool.map(partial_preprocess_and_clean, df_split))
#     pool.close()
#     pool.join()

#     # Save the filtered DataFrame to a CSV file
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = f"./filtered_data/{numClusters}_clusters_{timestamp}.csv"
#     df_processed.to_csv(filename, index=False)

#     return df_processed


# def preprocess_and_clean(cleanedColumnName: str, columnToClean: str, df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Preprocess and clean a chunk of DataFrame.

#     Args:
#         df (pd.DataFrame): Chunk of DataFrame to preprocess.

#     Returns:
#         pd.DataFrame: Preprocessed chunk of DataFrame.
#     """
#     df[cleanedColumnName] = df[columnToClean].apply(
#         lambda x: preprocess_text(x))
#     return df


# def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
#     """
#     Preprocess text by removing links, special characters, numbers, and optionally stopwords.

#     Args:
#         text (str): Input text.
#         remove_stopwords (bool): Whether to remove stopwords (default True).

#     Returns:
#         str: Preprocessed text.
#     """
#     # Lemmatization did not improve the results

#     text = re.sub(r"http\S+", "", text)  # Remove links
#     # text = re.sub("[^A-Za-z]+", " ", text)  # Remove special characters and numbers
#     if remove_stopwords:
#         tokens = nltk.word_tokenize(text)  # Tokenize
#         tokens = [w for w in tokens if not w.lower(
#         ) in stopwords.words("english")]  # Remove stopwords
#         text = " ".join(tokens)  # Join tokens
#     text = text.lower().strip()  # Convert to lowercase and remove whitespace
#     return text


def preprocess_data(data: pd.DataFrame, columnToClean: str, cleanedColumnName: str,
                    selected_subjects: list, numClusters) -> pd.DataFrame:
    """
    Filter it by selected subjects, and preprocess the text.

    Args:
        selected_subjects (list): List of subject areas to filter the data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """

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
    data[cleanedColumnName] = data[columnToClean].apply(
        lambda x: preprocess_text(x, remove_stopwords=True))

    # Save the filtered DataFrame to a CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./filtered_data/{numClusters}_clusters_{timestamp}.csv"

    data.to_csv(filename, index=False)

    return data


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
