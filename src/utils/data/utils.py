import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def preprop_data(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    df['Amount_log'] = np.log(df['Amount'] + 1e-9)
    df.drop(columns=['Amount', 'Time'], axis=1, inplace=True)
    data, target = df.drop(columns=['Class'], axis=1), df['Class']
    data_train, data_test, target_train, target_test = train_test_split(
        data.values,
        target.values,
        test_size=.2,
        stratify=target
    )

    return data_train, data_test, target_train, target_test
