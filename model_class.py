import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torchmetrics import AUROC, Accuracy, F1Score
import logging
from typing import Optional, List
from transformers import  BigBirdForSequenceClassification
from transformers import DistilBertForSequenceClassification


class FakeNewsClassifier(pl.LightningModule):
    def __init__(self, model_name: str,
                 device_type: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 initial_epochs: int = 0,
                 learning_rate: float = 1e-3,
                 class_weights: Optional[torch.Tensor] = None,
                 unfreeze_layers: Optional[List[str]] = None,
                 additional_fc: bool = False,
                 dropout_rate: Optional[float] = 0.5):
        super(FakeNewsClassifier, self).__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.device_type = device_type
        self.lr = learning_rate
        self.initial_epochs = initial_epochs
        self.unfreeze_layers_list = unfreeze_layers
        self.dropout = nn.Dropout(p=dropout_rate)
        self.class_weights = class_weights
        self.additional_fc = additional_fc

        if self.model_name == 'bigbird':
            self.feature_extractor = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base',
                                                                                      num_labels=1)
            num_features = self.feature_extractor.config.hidden_size
        elif self.model_name == 'distilbert':
            self.feature_extractor = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                                         num_labels=1)
            num_features = self.feature_extractor.config.hidden_size
        else:
            raise ValueError("Invalid model name")

        self.freeze_base_layers()

        if self.additional_fc:
            self.feature_extractor.classifier = nn.Sequential(
                nn.Linear(num_features, num_features // 2),
                nn.ReLU(),
                self.dropout
            )
            num_features = num_features // 2

        print("num_features after additional_fc:", num_features)

        self.classifier_head = nn.Linear(num_features, 1)

        self.feature_extractor.classifier.requires_grad = True
        self.classifier_head.requires_grad = True

        self.train_losses = []
        self.val_losses = []

        self.train_roc_auc = AUROC(num_classes=1, task='binary')
        self.val_roc_auc = AUROC(num_classes=1, task='binary')
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.val_f1 = F1Score(task='binary')

        self.current_epoch_counter = 0

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device_type)
        attention_mask = attention_mask.to(self.device_type)

        outputs = self.feature_extractor(input_ids=input_ids,
                                         attention_mask=attention_mask)
        logits = outputs.logits

        if self.additional_fc:
            logits = self.classifier_head(logits)
        else:
            logits = nn.Linear(logits.shape[-1], 1)(logits)

        return logits

    def on_train_epoch_start(self) -> None:
        """
        Callback called before each epoch.
        """
        if self.current_epoch < self.initial_epochs:
            self.freeze_base_layers()
            self.classifier_head.requires_grad = True
            logging.info("Freezing all base layers...")
        else:
            self.unfreeze_base_layers()
            self.classifier_head.requires_grad = True
            logging.info(f"Unfreezing layers in {self.unfreeze_layers_list} for fine-tuning...")

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Training for epoch {self.current_epoch} with {num_trainable_params} trainable parameters.")

    def training_step(self, batch, batch_idx: int):
        input_ids, attention_mask, true_labels = batch
        input_ids = input_ids.to(self.device_type)
        attention_mask = attention_mask.to(self.device_type)
        true_labels = true_labels.to(self.device_type)

        outputs = self(input_ids, attention_mask)
        if self.model_name == 'bigbird':
            true_labels = true_labels.unsqueeze(1)

        if self.additional_fc:
            logits = outputs[:, 0]
        else:
            logits = outputs

        if self.class_weights is not None:
            loss = F.binary_cross_entropy_with_logits(logits, true_labels.squeeze(1).float(), pos_weight=self.class_weights)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, true_labels.float())

        preds = torch.sigmoid(logits)
        binary_preds = (preds >= 0.5).float()

        roc_auc = self.train_roc_auc(preds, true_labels.long())
        accuracy_value = self.train_accuracy(binary_preds, true_labels.long())
        f1_value = self.train_f1(binary_preds, true_labels.long())

        self.log('train_loss', loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_acc', accuracy_value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_f1', f1_value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_roc_auc', roc_auc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return {'loss': loss, 'outputs': outputs, 'true_labels': true_labels}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids, attention_mask, true_labels = batch
            input_ids = input_ids.to(self.device_type)
            attention_mask = attention_mask.to(self.device_type)
            true_labels = true_labels.to(self.device_type)
            outputs = self(input_ids, attention_mask)
            if self.model_name == 'bigbird':
                true_labels = true_labels.unsqueeze(1)

            if self.additional_fc:
                logits = outputs[:, 0]
            else:
                logits = outputs

            loss = F.binary_cross_entropy_with_logits(logits, true_labels.float())

            preds = torch.sigmoid(logits)
            binary_preds = (preds >= 0.5).float()

            roc_auc = self.val_roc_auc(preds, true_labels.long())
            accuracy_value = self.val_accuracy(binary_preds, true_labels.long())
            f1_value = self.val_f1(binary_preds, true_labels.long())

            self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_acc', accuracy_value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_f1', f1_value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_roc_auc', roc_auc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {'loss': loss, 'outputs': outputs, 'true_labels': true_labels}

    def on_train_epoch_end(self) -> None:
        """
        Callback called after each training epoch.
        """
        avg_train_loss = self.trainer.callback_metrics['train_loss'].mean()
        avg_train_acc = self.trainer.callback_metrics['train_acc'].mean()
        avg_train_f1 = self.trainer.callback_metrics['train_f1'].mean()
        avg_train_roc_auc = self.trainer.callback_metrics['train_roc_auc'].mean()

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Epoch {self.current_epoch}. Trainable Parameters: {num_trainable_params}")
        logging.info(f"avg_train_loss: {avg_train_loss.item()}")
        print("avg_train_loss:", avg_train_loss.item())
        logging.info(f"avg_train_acc: {avg_train_acc.item()}")
        logging.info(f"avg_train_f1: {avg_train_f1.item()}")
        logging.info(f"avg_train_roc_auc: {avg_train_roc_auc.item()}")
        self.train_losses.append(avg_train_loss.item())

    def on_validation_epoch_end(self) -> None:
        """
        Callback called after each validation epoch.
        """
        if self.current_epoch_counter > 0:
            avg_val_loss = self.trainer.callback_metrics['val_loss'].mean()
            avg_val_acc = self.trainer.callback_metrics['val_acc'].mean()
            avg_val_f1 = self.trainer.callback_metrics['val_f1'].mean()
            avg_val_roc_auc = self.trainer.callback_metrics['val_roc_auc'].mean()

            logging.info(f"avg_val_loss: {avg_val_loss.item()}")
            print("avg_val_loss:", avg_val_loss.item())
            logging.info(f"avg_val_acc: {avg_val_acc.item()}")
            print("avg_val_acc:", avg_val_acc.item())
            logging.info(f"avg_val_f1: {avg_val_f1.item()}")
            print("avg_val_f1:", avg_val_f1.item())
            logging.info(f"avg_val_roc_auc: {avg_val_roc_auc.item()}")
            print("avg_val_roc_auc:", avg_val_roc_auc.item())
            self.val_losses.append(avg_val_loss.item())
        else:
            logging.info("No validation metrics for this epoch.")
        self.current_epoch_counter += 1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss'
            }
        }

    def freeze_base_layers(self):
        if self.unfreeze_layers_list is not None:
            for name, param in self.feature_extractor.named_parameters():
                if all([not name.startswith(layer) for layer in self.unfreeze_layers_list]):
                    param.requires_grad = False
        else:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def unfreeze_base_layers(self) -> None:
        """
        Unfreeze the parameters of the specified layers in the base feature extractor.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        if self.unfreeze_layers_list is not None:
            for name, child in self.feature_extractor.named_children():
                if name in self.unfreeze_layers_list:
                    for params in child.parameters():
                        params.requires_grad = True