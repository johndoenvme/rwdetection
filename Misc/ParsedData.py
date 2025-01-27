import pandas as pd
from typing import Any


class ParsedData:

    def __init__(self, label: Any, data: pd.DataFrame, train_test: str, rec_number: int,
                 metadata: Any = None):  # Label could be int (e.g for Trees) or even vectors (e.g for CLT)
        self.label = label
        self.data = data
        self.train_test = train_test
        self.rec_number = rec_number
        self.metadata = metadata
