import numpy as np
import torch
from torch import optim
import pandas as pd
from tqdm import tqdm
from typing import List
from Models.CLTModel import CLTModel
from Models.PLTModel import PLTModel
from Misc.ModelInput import ModelInput
from Misc.TransformerDataset import TransformerDataset
from Misc import Utils
from torch.utils.data import DataLoader

reduce_factor = 66


def transformer_collate_fn(batch):
    return batch


class TransformerManager:

    def __init__(self, model_name: str, dataset: List[ModelInput], demo_mode: bool,
                 learning_rate: float = 0.0001,
                 num_epochs=200,
                 ):
        self.demo_mode = demo_mode
        self.dataset = dataset
        self.dataset_train_, self.dataset_test_ = Utils.divide_train_test(dataset)
        self.dataset_train = TransformerDataset(self.dataset_train_)
        self.dataset_test = TransformerDataset(self.dataset_test_)

        # set model & schedule the optimizer
        self.model_name = model_name
        if model_name == 'CLT':
            scheduler = {'step': 30, 'gamma': 0.8}
            self.model = CLTModel()
            self.batch_size = 64
        else:
            scheduler = {'step': 5000, 'gamma': 0.8}
            self.model = PLTModel()
            self.batch_size = 256
        self.num_epochs = num_epochs
        self.test_size = 0.67  # For splits
        self.num_splits = 50
        if self.demo_mode:
            self.batch_size = 2
            self.num_epochs = 1
            self.num_splits = 1
            self.test_size = 0.5


        self.dataset_train_loader = DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, collate_fn=transformer_collate_fn)
        self.dataset_test_loader = DataLoader(dataset=self.dataset_test, batch_size=self.batch_size, collate_fn=transformer_collate_fn)
        self.num_train_batches = self.dataset_train_.__len__() // self.batch_size
        self.num_test_batches = self.dataset_test_.__len__() // self.batch_size
        self.epoch_num = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = learning_rate
        self.loss = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler['step'],
                                                   gamma=scheduler['gamma'])
        self.reduce_factor = reduce_factor if model_name == "CLT" else 1  # factor 1 -> no reduction

    def train(self):
        # Loop for each Epoch
        while self.epoch_num < self.num_epochs:
            # Perform a training iteration
            self._training_iteration_per_db()

            # # Check if it's time to perform a validation step.
            # if self.epoch_num % self.validation_every_n_epochs == 0:
            #     validation_loss, validation_auc = self._eval_validation_step()

            self.scheduler.step()
            self.epoch_num += 1

    def inference(self):
        chunk_results = self._evaluation_iteration()

        # Benchmark results
        results_dict = Utils.benchmark_results(results_input=chunk_results, num_splits=self.num_splits, test_size=self.test_size)

        return results_dict

    def _training_iteration_per_batch(self, batch: ModelInput) -> torch.Tensor:
        """
        """
        states_, labels_ = Utils.obtain_features_and_labels(batch)
        states = torch.Tensor(states_).to(self.device)
        labels = torch.Tensor(labels_).to(self.device)

        res = self.model(states).squeeze()
        if self.model_name == 'PLT':
            labels = labels[:, :2, :]
        loss = self.loss(res, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _training_iteration_per_db(self):
        """
        """
        # Create DatasetLogic Iteration
        ds_iter = iter(self.dataset_train_loader)

        # Keep track of our losses
        losses = []

        # Loop per batch in our dataset
        for _ in tqdm(range(self.num_train_batches)):
            curr_batch = next(ds_iter)

            loss = self._training_iteration_per_batch(curr_batch)
            losses.append(loss.cpu().detach().numpy())

    def _evaluation_iteration(self):

        y_preds, y_targs = [], []

        with torch.no_grad():
            for curr_batch in tqdm(self.dataset_test, total=self.num_test_batches):
                states_, labels_ = Utils.obtain_features_and_labels(curr_batch)
                states = torch.Tensor(states_).to(self.device)
                if labels_.ndim <= 2:
                    labels_ = labels_[np.newaxis, :]
                if states.ndimension() <= 2:
                    states = states.unsqueeze(0)
                if self.model_name == 'PLT':
                    labels_ = labels_[:, :2, :]

                res = self.model(states).detach().cpu().numpy()

                y_targs.append(Utils.fracreg_to_label(labels_, self.model_name).tolist())
                y_preds.append(Utils.fracreg_to_predict_prob(res, self.model_name).tolist())
                rec_idx = [x.parsed_data_element.rec_number for x in self.dataset_test]

        results = pd.DataFrame({'target': y_targs, 'pred_prob': y_preds, 'rec_idx': rec_idx})

        return results
