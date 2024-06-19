from pandas import DataFrame
from PIL import UnidentifiedImageError
from tensorflow.keras.preprocessing import image


def is_valid_image(path: str) -> bool:
    """
    Checks if an image at the specified path is valid and can be loaded.

    Args:
        path (str): Path to the image file.

    Returns:
        bool: True if the image is valid and can be loaded, False otherwise.
    """
    try:
        image.load_img(path)
        return True
    except UnidentifiedImageError:
        return False


def clean_and_preprocess_dataframe(df: DataFrame) -> DataFrame:
    """
    Cleans and preprocesses the dataframe containing image URLs.

    Args:
        df (DataFrame): DataFrame with columns including 'url' containing image URLs.

    Returns:
        DataFrame: Processed DataFrame with an additional 'local_path' column containing local paths of valid images.
    """

    df["local_path"] = df["url"].str.split("/").apply(lambda x: f"images/{x[-1]}")
    df = df[df.local_path.apply(is_valid_image)]
    return df
