import pandas as pd

from sklearn.linear_model import LogisticRegression

from utils.logging import get_logger


logger = get_logger(name=__name__)

def learn_linear_model(
    data_train: pd.DataFrame,
    target_train: pd.Series
) -> LogisticRegression:
    logger.info('Learning Logistic Regression')
    lg = LogisticRegression(C=.74, warm_start=True)
    lg.fit(data_train, target_train)
    logger.info('Logistic Regression fitted!')

    return lg
