import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


def process_image(img: Image.Image) -> np.ndarray:
    """
    Preprocesses the image for the ResNet model.

    Args:
        img (Image.Image): The image to process.

    Returns:
        np.ndarray: The preprocessed image array.
    """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_embedding_from_path(img_path: str, model: tf.keras.models.Model) -> np.ndarray:
    """
    Generates the embedding for the input image using the provided model.

    Args:
        img_path (str): Path to the image file.
        model (tf.keras.models.Model): The model used to generate the embedding.

    Returns:
        np.ndarray: The flattened embedding vector.
    """
    # Load and preprocess an image
    img = image.load_img(img_path, target_size=(224, 224))
    x = process_image(img)

    # Generate embeddings
    embedding = model.predict(x, verbose=0)

    return embedding.flatten()


def get_embeddings_dataframe(df: pd.DataFrame, model: tf.keras.models.Model) -> pd.DataFrame:
    """
    Generates embeddings for all images in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing image paths.
        model (tf.keras.models.Model): The model used to generate embeddings.

    Returns:
        pd.DataFrame: Dataframe with embeddings for all images and their IDs.
    """
    # Get embedding for each image
    embeddings_serie = df["local_path"].apply(get_embedding_from_path, model=model)

    # Organize the embeddings of images in a DataFrame with their IDs
    embeddings_df = pd.DataFrame(embeddings_serie.to_list(), index=df.index)
    return embeddings_df


if __name__ == "__main__":
    from tensorflow.keras.applications import ResNet50V2

    from utils import clean_and_preprocess_dataframe

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help="input CSV file with image URLs", default="data.csv")
    parser.add_argument("--output_csv_file", help="output CSV file with image embeddings", default="embeddings.csv")
    args = parser.parse_args()

    # Read and preprocess the input CSV file
    df = pd.read_csv(args.csv_file, index_col="id")
    df = clean_and_preprocess_dataframe(df)

    # Load the pre-trained ResNet50V2 model without classification layers
    model = ResNet50V2(weights="imagenet", include_top=False, pooling="avg")

    # Generate embeddings and save to the output CSV
    embeddings_df = get_embeddings_dataframe(df, model)
    embeddings_df.to_csv(args.output_csv_file)
