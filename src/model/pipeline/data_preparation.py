"""
This module provides functionality for preparing a dataset for ML model.

It consists of functions to load data from a database,
encode categorical columns, and parse specific columns for further processing.
"""

import pandas as pd
from loguru import logger

from model.pipeline.data_collection import load_data_from_db


def prepare_data() -> pd.DataFrame:
    """
    Prepare the dataset for analysis and modelling.

    This involves loading the data, encoding categorical columns,
    and parsing the 'garden' column.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    logger.info('starting up preprocessing pipeline')
    dataframe = load_data_from_db()
    data_encoded = _encode_cat_columns(dataframe)
    return _parse_garden_column(data_encoded)


def _encode_cat_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns into dummy variables.

    Arg:
        dataframe (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: Dataset with categorical columns encoded.
    """
    cols = ['balcony', 'parking', 'furnished', 'garage', 'storage']
    logger.info(f'Encoding columns: {cols}')
    return pd.get_dummies(
        dataframe,
        columns=cols,
        drop_first=True,
        )


def _parse_garden_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'garden' column in the dataset.

    Args:
        dataframe (pd.DataFrame): The dataset with a 'garden' column.

    Returns:
        pd.DataFrame: The dataset with the 'garden' column parsed.
    """
    dataframe['garden'] = (dataframe.garden
                                    .str
                                    .extract(r'(\d+)')  # extract the numbers.
                                    .astype('float')  # cast as float.
                                    .fillna(0))  # fill nulls as 0.
    return dataframe
