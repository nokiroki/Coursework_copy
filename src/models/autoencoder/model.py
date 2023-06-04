from typing import Any, Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn


class AutoEncoder(LightningModule):

    def __init__(
        self,
        in_out_features: int,
        hidden_size: int,
        learning_rate: float,
        weight_decay: float,
        dropout: float = .2,
        *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters({
            'in_out_features': in_out_features,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout': dropout
        })

        self.encoder = nn.Sequential(
            nn.Linear(in_out_features, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(128, hidden_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(512, in_out_features)
        )

        self.criterion = nn.MSELoss()

        self.pred_losses = []

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data_encoded = self.encoder(data)
        data_decoded = self.decoder(data_encoded)

        return data_decoded
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data, target = batch
        data_restored = self(data)

        loss = self.criterion(data_restored, data)

        self.pred_losses.append((target.cpu().item(), loss.cpu().item()))

        return loss

    def training_step(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        data, _ = batch
        data_restored = self(data)

        loss = self.criterion(data_restored, data)

        self.log('train_loss', loss, prog_bar=True, on_step=True)

        return loss
    
    def validation_step(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> None:
        data, _ = batch
        data_restored = self(data)

        loss = self.criterion(data_restored, data)

        self.log('val_loss', loss, prog_bar=True, on_step=True)
    
    def test_step(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> None:
        data, _ = batch
        data_restored = self(data)

        loss = self.criterion(data_restored, data)

        self.log('test_loss', loss, prog_bar=True, on_step=True)
    
    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(
            self.parameters(),
            self.hparams['learning_rate'],
            weight_decay=self.hparams['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'min', 1e-1, 2, verbose=True
        )
        return [opt], [{'scheduler': scheduler, 'monitor': 'val_loss'}]
