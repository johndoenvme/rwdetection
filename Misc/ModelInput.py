import numpy as np
from Misc.ParsedData import ParsedData


class ModelInput:

    def __init__(self, parsed_data_element: ParsedData, features: np.ndarray):
        self.parsed_data_element = parsed_data_element
        self.features = features
