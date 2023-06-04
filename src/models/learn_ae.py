import os

import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from autoencoder import FraudDataModule, AutoEncoder


if __name__ == '__main__':

    dataset = pd.read_csv(
        os.path.join('..', '..', 'data', 'creditcard.csv')
    )

    data_module = FraudDataModule(
        dataset, 128, 128, 129, 4, drop_positive=True, target_name='Class'
    )

    model = AutoEncoder(dataset.shape[1] - 2, 16, 1e-3, 1e-5)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=.01,
        patience=5,
        verbose=True,
        mode='min'
    )

    checkpoint = ModelCheckpoint(
        os.path.join('..', '..', 'logs', 'ae', 'checkpoints'),
        'autoencoder',
        'val_loss',
        mode='min'
    )

    callbacks = [early_stopping, checkpoint]

    logger = TensorBoardLogger(
        os.path.join('..', '..', 'logs', 'ae', 'logs'),
        'autoencoder'
    )

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        log_every_n_steps=20,
        logger=logger,
        callbacks=callbacks,
        max_epochs=100
    )

    trainer.fit(model, datamodule=data_module)
