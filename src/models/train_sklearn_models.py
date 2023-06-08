from typing import Callable

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from utils.logging import get_logger


logger = get_logger(name=__name__)

def name_funс(name: str):
    def decorator(func: Callable):
        def wrapper(*args,  **kwargs):
            logger.info(f'Fitting{name}')
            result = func(*args, **kwargs)
            logger.info(f'{name} fitted!')
            return result
        return wrapper
    return decorator


@name_funс('Logistic Refression')
def learn_linear_model(
    data_train: pd.DataFrame,
    target_train: pd.Series
) -> LogisticRegression:
    lg = LogisticRegression(C=.74, warm_start=True, n_jobs=-1)
    lg.fit(data_train, target_train)

    return lg


@name_funс('Oversampling')
def learn_oversampling_model(
    data_train: pd.DataFrame,
    target_train: pd.Series
)-> LogisticRegression:
    data_train_resampled, target_train_resampled = SMOTE().fit_resample(
        data_train, target_train
    )

    lg = LogisticRegression(C=.74, warm_start=True, n_jobs=-1)
    lg.fit(data_train_resampled, target_train_resampled)

    return lg

@name_funс('Elliptic Envelope')
def learn_envelope(data_train: pd.DataFrame) -> EllipticEnvelope:
    envelope = EllipticEnvelope(contamination=.01)
    envelope.fit(data_train)

    return envelope


@name_funс('RandomForest')
def learn_random_forest(
    data_train: pd.DataFrame,
    target_train: pd.Series
)-> RandomForestClassifier:
    ran_for = RandomForestClassifier(
        n_jobs=-1,
        n_estimators=500
    )
    ran_for.fit(data_train, target_train)

    return ran_for
