import pickle as pk

from loguru import logger

from model.pipeline.model_training import build_model
from config import model_settings


class ModelService:
    """
    A service class for managing the ML model.

    This class provides functionalities to load a ML model from
    a specified path, build it if it doesn't exist, and make
    predictions using the loaded model.

    Attributes:
        model: ML model managed by this service. Initially set to None.

    Methods:
        __init__: Constructor that initializes the ModelService.
        load_model: Loads the model or builds it if it doesn't exist.
        predict: Makes a prediction using the loaded model.
    """

    def __init__(self):
        self.model = None

    def load_model(self):
        path = model_settings.model_path / model_settings.model_name

        if not path.exists():
            logger.warning(f'Model {model_settings.model_name} not found.'
                           f'Building the model')
            build_model()

        logger.info(f'Model loaded from {path}')

        with open(path, 'rb') as model_file:
            self.model = pk.load(model_file)

    def predict(self, input_parameters: list) -> list:
        """
        Makes a prediction using the loaded model.

        Takes input parameters and passes it to the model, which
        was loaded using a pickle model_file:.

        Args:
            input_parameters (list): The input data for making a prediction.

        Returns:
            list: The prediction result from the model.
        """

        return self.model.predict([input_parameters])
