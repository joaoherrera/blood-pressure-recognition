""" A basic model for machine learning.
"""

import numpy as np
import torch

from learning.base import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    EPOCHS = 100

    def train(self, dataset, optimizer, loss_func):
        loss_training = []

        # Set module status to training. Implemented in torch.nn.Module
        self.model.train()

        with torch.set_grad_enabled(True):
            for batch in dataset:
                x_pred, y_true = batch
                x_pred, y_true = x_pred.to(self.device), y_true.to(self.device)

                # Predict
                y_pred = self.model(x_pred)

                # Loss computation and weights correction
                loss = loss_func(y_pred, y_true)
                loss.backward()  # backpropagation
                optimizer.step()

                loss_training.append(loss.item())
        return np.mean(loss_training)

    def evaluate(self, dataset, coef_func):
        coef_validation = []

        # Set module status to evalutation. Implemented in torch.nn.Module
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                x_pred, y_true = batch
                x_pred, y_true = x_pred.to(self.device), y_true.to(self.device)

                # Predict
                y_pred = self.model(x_pred)

                coef = coef_func(y_pred, y_true)
                coef_validation.append(coef.item())
        return np.mean(coef_validation)

    def fit(self, training_dataset, validation_dataset, optimizer, loss_func, coef_func):
        for epoch in range(self.EPOCHS):
            print(f"Epoch {epoch}")

            loss_training = self.train(training_dataset, optimizer, loss_func)
            coef_evalutation = self.evaluate(validation_dataset, coef_func)

            print(f"Loss training: {loss_training}")
            print(f"Coefficient evaluation: {coef_evalutation}")
