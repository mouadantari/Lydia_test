import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances


def closest_image_id(input_embedding: pd.DataFrame, embeddings_df: pd.DataFrame, n: int = 3) -> list[int]:
    """
    Finds the IDs of the closest images to the input embedding.

    Args:
        input_embedding (np.ndarray): The embedding of the input image.
        embeddings_df (pd.DataFrame): DataFrame containing embeddings of images in the dataset.
        n (int, optional): Number of closest images to return. Defaults to 3.

    Returns:
        list[int]: List of IDs of the closest images.
    """
    # Compute cosine distances between the input embedding and all embeddings in the DataFrame
    distances = cosine_distances(np.expand_dims(input_embedding, 0), embeddings_df).flatten()

    # Get the indices of the n smallest distances
    min_distances_arg = distances.argsort()[:n]

    # Return the IDs of the n closest images
    return embeddings_df.index[min_distances_arg].to_list()
