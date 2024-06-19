import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.applications import ResNet50V2

from src.embeddings import get_embedding_from_path
from src.inference import closest_image_id
from src.utils import is_valid_image

app = FastAPI()


class ImageRequest(BaseModel):
    """Model to represent the image path in the request body."""

    image_path: str


@app.get(
    "/ping",
    response_model=str,
    summary="Ping endpoint to check if the server is running",
)
def ping() -> str:
    """
    A simple GET endpoint to check if the server is running.

    Returns:
        str: Returns "pong" if the server is running.
    """
    return "pong"


@app.post(
    "/closest_img_ids",
    response_model=list[int],
    summary="POST endpoint that takes an image path an returns the IDs of the closest images to the input",
)
def get_closest_img_ids(request: ImageRequest) -> list[int]:
    """
    POST endpoint that takes an image path and returns the IDs of the closest images in the dataset.

    Args:
        request (ImageRequest): The request body containing the image path.

    Returns:
        list[int]: A list of IDs of the closest images.

    Raises:
        HTTPException: If the image path is not valid or an error occurs during processing.
    """
    # Validate the image path
    if not is_valid_image(request.image_path):
        raise HTTPException(status_code=400, detail="Image not valid")
    try:
        # Load the pre-trained ResNet50V2 model
        model = ResNet50V2(weights="imagenet", include_top=False, pooling="avg")

        # Generate the embedding for the input image
        input_embedding = get_embedding_from_path(request.image_path, model)

        # Load the embeddings from the CSV file
        embeddings_df = pd.read_csv("embeddings.csv", index_col="id")

        # Get the closest image IDs
        return closest_image_id(input_embedding, embeddings_df)

    except Exception as e:
        # Exceptions handling
        raise HTTPException(status_code=500, detail=str(e))
