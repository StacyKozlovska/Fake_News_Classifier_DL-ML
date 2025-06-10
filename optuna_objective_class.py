import logging
from torch.utils.data import DataLoader
from model_class import FakeNewsClassifier
import pytorch_lightning as pl

class Objective:
    def __init__(self, train_dataset, val_dataset, model_name, initial_epochs=0,
                 class_weights=None, additional_fc=True, unfreeze_layers=None,
                 max_epochs=10, dropout_rate=0.5):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_name = model_name
        self.class_weights = class_weights
        self.additional_fc = additional_fc
        self.unfreeze_layers = unfreeze_layers
        self.initial_epochs = initial_epochs
        self.max_epochs = max_epochs
        self.dropout_rate = dropout_rate

    def __call__(self, trial):
        try:
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

            logging.info(f"Trial: batch_size={batch_size}, learning_rate={learning_rate}")

            train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size)

            model = FakeNewsClassifier(
                model_name=self.model_name,
                additional_fc=self.additional_fc,
                class_weights=self.class_weights,
                unfreeze_layers=self.unfreeze_layers,
                initial_epochs=self.initial_epochs,
                dropout_rate=self.dropout_rate,
                learning_rate=learning_rate,
            )

            trainer = pl.Trainer(max_epochs=self.max_epochs, logger=True, enable_checkpointing=False)

            trainer.fit(model, train_dataloader, val_dataloader)

            val_loss = trainer.callback_metrics['val_loss'].mean().item()
            logging.info(f"Validation Loss: {val_loss:.4f}")

            return val_loss
        except Exception as e:
            logging.error(f"Error occurred during optimization: {e}")
            return 1e9 