import numpy as np
import pandas as pd

from src.inference import closest_image_id
from src.utils import clean_and_preprocess_dataframe, is_valid_image


def test_is_valid_image():
    assert not is_valid_image("images/406453588_b2541371b8_o.jpg")
    assert is_valid_image("images/129983132_b668be4a47_o.jpg")


def test_clean_and_preprocess_dataframe():
    df = pd.DataFrame.from_dict(
        {
            "url": {
                75680: "https://farm4.staticflickr.com/177/406453588_b2541371b8_o.jpg",
                461828: "https://c4.staticflickr.com/9/8555/15625756039_a60b0bd0a5_o.jpg",
            }
        }
    )
    assert (
        clean_and_preprocess_dataframe(df)
        == pd.DataFrame.from_dict(
            {
                "url": {461828: "https://c4.staticflickr.com/9/8555/15625756039_a60b0bd0a5_o.jpg"},
                "local_path": {461828: "images/15625756039_a60b0bd0a5_o.jpg"},
            }
        )
    ).all(axis=None)


def test_closest_image_id():
    embeddings_df = pd.DataFrame(np.random.rand(2, 10))
    input_embedding = embeddings_df.iloc[[0]]
    embeddings_df = pd.concat([input_embedding, input_embedding, embeddings_df]).reset_index(drop=True)
    assert closest_image_id(input_embedding.iloc[0], embeddings_df) == [0, 1, 2]
