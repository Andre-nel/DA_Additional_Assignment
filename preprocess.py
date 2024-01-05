from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from typing import List
from multiprocessing import Pool

# Precompile regular expressions and load stopwords
url_regex = re.compile(r"http\S+")
stop_words = set(stopwords.words("english"))


def preprocessText(text, removeStopwords: bool = True):
    """
    Preprocesses text by removing links, special characters, numbers, and optionally stopwords.

    Args:
        text: Input text. Can be any type, but expected to be a string most of the time.
        removeStopwords (bool): Flag to indicate whether to remove stopwords (default is True).

    Returns:
        str: Preprocessed text. Returns an empty string if input is not a string.
    """
    # Check if the text is not a string (e.g., float, int)
    if not isinstance(text, str):
        return str(text)

    text = url_regex.sub("", text)  # Remove links
    if removeStopwords:
        tokens = nltk.word_tokenize(text)  # Tokenize
        # Remove stopwords
        tokens = [w for w in tokens if not w.lower() in stop_words]
        text = " ".join(tokens)  # Join tokens
    return text.lower().strip()  # Convert to lowercase and remove whitespace


# def parallelPreprocess(data: pd.DataFrame, column: str) -> pd.Series:
#     """
#     Apply preprocessing in parallel.

#     Args:
#         data (pd.DataFrame): DataFrame containing the text data.
#         column (str): Column name of the text data.

#     Returns:
#         pd.Series: Preprocessed text data.
#     """
#     with Pool() as pool:
#         return pd.Series(pool.map(preprocessText, data[column]))


# def preprocessData(data: pd.DataFrame, columnToClean: str, cleanedColumnName: str, numClusters: int) -> pd.DataFrame:
#     """
#     Preprocesses the text data in a specified column of a DataFrame.

#     The function applies text cleaning (like removing links, special characters, stopwords),
#     and saves the cleaned data into a new column. Additionally, it saves the DataFrame
#     to a CSV file named based on the number of clusters and the current timestamp.

#     Args:
#         data (pd.DataFrame): The DataFrame containing text data.
#         columnToClean (str): The name of the column to be cleaned.
#         cleanedColumnName (str): The name of the column to store the cleaned data.
#         numClusters (int): The number of clusters used in the filename for saving the DataFrame.

#     Returns:
#         pd.DataFrame: The DataFrame with the cleaned text data.
#     """
#     if columnToClean not in data.columns:
#         raise ValueError(f"Column '{columnToClean}' not found in DataFrame")

#     # Apply text preprocessing in parallel
#     data[cleanedColumnName] = parallelPreprocess(data, columnToClean)

#     # Save the DataFrame to a CSV file
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = f"./filtered_data/{numClusters}_{data.shape[0]}_ppd_clusters_{timestamp}.csv"
#     data.to_csv(filename, index=False)

#     return data


def preprocessData(data: pd.DataFrame,
                   columnToClean: str,
                   cleanedColumnName: str,
                   numClusters: int,
                   batchSize: int = 10000) -> pd.DataFrame:
    """
    Preprocess the text data in a specified column of a DataFrame with batching.

    Args:
        data (pd.DataFrame): The DataFrame containing text data.
        columnToClean (str): The name of the column to be cleaned.
        cleanedColumnName (str): The name of the column to store the cleaned data.
        numClusters (int): The number of clusters used in the filename for saving the DataFrame.
        batchSize (int): The size of each batch for processing. Default is 10,000.

    Returns:
        pd.DataFrame: The DataFrame with the cleaned text data.
    """
    if columnToClean not in data.columns:
        raise ValueError(f"Column '{columnToClean}' not found in DataFrame")

    # Function to process each batch
    def processBatch(batch):
        with Pool(cpu_count()) as pool:
            return pool.map(preprocessText, batch)

    # Split the DataFrame into batches and process each batch
    batches = np.array_split(
        data[columnToClean], np.ceil(len(data) / batchSize))
    processed_batches = [processBatch(batch) for batch in batches]

    # Concatenate the processed batches and assign to the DataFrame
    data[cleanedColumnName] = pd.concat(
        [pd.Series(batch) for batch in processed_batches], ignore_index=True)

    # Save the DataFrame to a CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"./filtered_data/{numClusters}_{data.shape[0]}_ppd_clusters_{timestamp}.csv"
    data.to_csv(filename, index=False)

    return data
