import pandas as pd
from typing import List
from Misc.ModelInput import ModelInput
from Models.RFModel import RFModel
from Models.DeftPunkModel import DeftPunkModel
from Misc import Utils
import numpy as np


class TreeManager:

    def __init__(self,
                 model_name: str,
                 dataset: List[ModelInput],
                 demo_mode: bool
                 ):

        if model_name == 'RF':
            self.model = RFModel()
        else:
            self.model = DeftPunkModel()

        self.dataset_train, self.dataset_test = Utils.divide_train_test(dataset)
        self.X_train, self.y_train = Utils.obtain_features_and_labels(self.dataset_train)
        self.X_test, self.y_test = Utils.obtain_features_and_labels(self.dataset_test)
        self.demo_mode = demo_mode

        if self.demo_mode:
            self.num_splits = 1
            self.test_size = 0.5
        else:
            self.num_splits = 50
            self.test_size = 0.67

    def train(self):
        self.ML_model = self.model.get_pipeline()
        self.ML_model.fit(self.X_train, self.y_train)

    def inference(self):
        rec_idx = [x.parsed_data_element.rec_number for x in self.dataset_test]
        results = pd.DataFrame({'pred_prob': self.ML_model.predict(self.X_test)})
        results['target'] = self.y_test
        results['rec_idx'] = rec_idx

        results_dict = Utils.benchmark_results(results_input=results, num_splits=self.num_splits, test_size=self.test_size)

        return results_dict
