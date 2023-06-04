from typing import Any, Optional

import numpy as np
import pandas as pd

import torch

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import Dataset, DataLoader, random_split

from pytorch_lightning import LightningDataModule


class FraudDataset(Dataset):

    def __init__(self, dataset: pd.DataFrame, target_name: str = 'class') -> None:
        super().__init__()
        self.target_name = target_name
        self.dataset = dataset

    def __getitem__(self, index: int) -> Any:
        values_target = self.dataset.iloc[index]
        values = values_target.drop(self.target_name).values
        target = values_target[self.target_name]
        values = torch.Tensor(values).float()

        return values, target
    
    def __len__(self) -> int:
        return self.dataset.shape[0]


class FraudDataModule(LightningDataModule):

    def __init__(
        self,
        data_train_val: pd.DataFrame,
        train_batch_size: int,
        val_batch_size:int,
        test_batch_size: int,
        num_workers: int,
        data_test: Optional[pd.DataFrame] = None,
        target_name: str = 'class',
        drop_positive: bool = True
    ) -> None:
        super().__init__()
        self.target_name = target_name
        self.data_train_val = data_train_val
        self.data_test = data_test
        self.drop_positive = drop_positive

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train: FraudDataset       = None
        self.val: FraudDataset         = None
        self.test: FraudDataset        = None

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.data_train_val = self._preprocessing(
                self.data_train_val, self.drop_positive
            )
            train_val = FraudDataset(self.data_train_val, self.target_name)
            length = len(train_val)
            self.train, self.val = random_split(
                train_val,
                (length - int(length * .2), int(length * .2))
            )
        elif stage == 'test':
            if self.data_test is None:
                raise Exception('None')
            self.data_test = self._preprocessing(
                self.data_test, False
            )
            self.test = FraudDataset(self.data_test, self.target_name)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers
        )


    def _preprocessing(self, df: pd.DataFrame, drop_positive: bool) -> pd.DataFrame:
        df['Amount_log'] = np.log(df['Amount'] + 1e-9)
        df.drop(columns=['Amount', 'Time'], axis=1, inplace=True)
        if drop_positive:
            df.drop(index=df[df[self.target_name] == 1].index, inplace=True)
        return df
