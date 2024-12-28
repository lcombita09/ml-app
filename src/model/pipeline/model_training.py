import pickle as pk

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple

from config import model_settings
from model.pipeline.data_preparation import prepare_data


def build_model():
    """
    Build, evaluate and save a RandomForestRegressor model.

    This function orchestrates the model building pipeline.
    It starts by preparing the data, followed by defining deature names
    and splitting the dataset into features and target variables.
    The dataset is then divided into training and testing sets.
    The model's performance is evaluated on the test set, and
    finally, the model is saved for future use.

    Return:
        None
    """
    logger.info('Building the model')
    # 1. Prepare the data
    df = prepare_data()
    feature_names = [
        'area',
        'constraction_year',
        'bedrooms',
        'garden',
        'balcony_yes',
        'parking_yes',
        'furnished_yes',
        'garage_yes',
        'storage_yes',
    ]
    # 2. Get the features and target variables
    x, y = get_x_y(
        df,
        col_x=feature_names,
        )
    # 3. Split the data into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,  # noqa: WPS432
        random_state=42,  # noqa: WPS432
        )
    # 4. Train the model
    rf = model_train(
        x_train,
        y_train,
        )
    # 5. Evaluate the model
    model_evaluate(
        rf,
        x_test,
        y_test,
        )
    # 6. Save the model
    save_model(rf)


def get_x_y(
    dataframe: pd.DataFrame,
    col_x: List[str],
    col_y: str = 'rent',
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features and target variable.

    Args:
        data (pd.DataFrame): The dataset to be split.
        col_x (list[str]): List of column names for features.
        col_y (str): Name of the target variable column.

    Returns:
        tuple: Features and target variables.
    """
    logger.info(f'Features: {col_x}')
    return dataframe[col_x], dataframe[col_y]


def model_train(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestRegressor:
    """
    Train the RandomForestRegressor model with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training set features.
        y_train (pd.Series): Training set target.

    Returns:
        RandomForestRegressor: The best estimator after GridSearch.
    """
    logger.info('Training the model')
    grid_space = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9, 12],
        }

    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid=grid_space,
        cv=5,
        scoring='r2',
        )

    model_grid = grid.fit(
        x_train,
        y_train,
        )

    return model_grid.best_estimator_


def model_evaluate(
    model: RandomForestRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """
    Evaluate the trained model's performance.

    Args:
        model (RandomForestRegressor): The trained model.
        X_test (pd.DataFrame): Testing set features.
        y_test (pd.Series): Testing set target.

    Returns:
        float: The model's score.
    """
    score = model.score(
        x_test,
        y_test,
        )
    logger.info(f'Evaluating the model. SCORE = {score}')
    return score


def save_model(model: RandomForestRegressor) -> None:
    """
    Save the trained model to a specified directory.

    Args:
        model (RandomForestRegressor): The model to save.

    Return:
        None
    """
    path = model_settings.model_path / model_settings.model_name
    logger.info(f'Saving the model to {path}')
    with open(path, 'wb') as f:
        pk.dump(model, f)
