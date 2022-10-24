"""
Source code for Splitter module.
Module splits DataFrame into 3 parts according to provided partition sizes.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from source.exception import StratifiedClassHasOneMember, ErrorTrainTestSplit


def train_test_split_exc(df_src, train_size, random_state, stratify, stage=''):
    """
    Raises Exception with user-friendly description

    :param stage: message for user when an error was occurred
    :return: df_train, df_test
    """
    if stratify is not None and stratify.value_counts().min() < 2:
        raise StratifiedClassHasOneMember(stage) from None

    try:
        df_train, df_test = train_test_split(
            df_src,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify
        )
        return df_train, df_test

    except Exception as exc:
        raise ErrorTrainTestSplit(stage, str(exc)) from None


def convert_to_float(size, divider):
    """
    Scales size if size's type is integer due to divider(df_src size)
    """
    if isinstance(size, int) and size >= 1:
        size /= divider
    return size


def splitter(df_src, train_size, test_size=None, seed=None,
             split_method="random", stratification_column=None, **kwargs):
    """
    Splitter main function.

    :param df_src: input DataFrame.
    :param train_size: size of the train part.
    :param test_size: size of test part.
    :param seed: random state for split generation.
    :param split_method: split method, "random" or "random_with_stratification"
    :param stratification_column: the name of a column for stratification.
    :return: a dict with train, test and validation subsets.
    """

    # Prepare train_size as float
    train_size = convert_to_float(train_size, len(df_src))

    # Prepare test_size as float
    if test_size is None:
        test_size = 1 - train_size
 
    test_size = convert_to_float(test_size, len(df_src))

    if len(df_src) == 0:
        raise ValueError('Empty DataFrame.')

    if (train_size >= 1) or (train_size < 0):
        raise ValueError('train_size={} should be smaller than 1.0 and higher '
                         'or equal than 0.'.format(train_size))

    if (test_size >= 1) or (test_size < 0):
        raise ValueError('val_size={} should be smaller than 1.0 and higher or '
                         'equal than 0.'.format(test_size))

    if train_size + test_size > 1:
        raise ValueError('Incorrect split sizes, train_size+test_size > 1.')

    if seed is not None:
        seed = int(seed) & 0xffffffff

    stratify = (None if stratification_column is "" or split_method == "random"
                else df_src[stratification_column])

    # Call of our custom function with exceptions catching
    df_train, val_and_test_df = train_test_split_exc(
        df_src,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
        stage='Error during splitting input DataFrame. '
    )

    # Creating DataFrame for validation
    if train_size + test_size < 1:
        stratify = (
            None if stratification_column is "" or split_method == "random"
            else val_and_test_df[stratification_column])

        df_test, df_val = train_test_split_exc(
            val_and_test_df,
            train_size=test_size/(1-train_size),
            random_state=seed,
            stratify=stratify,
            stage='Error during creating validation DataFrame. '
        )
    else:
        df_test = val_and_test_df
        df_val = pd.DataFrame(columns=df_test.columns)

    return {'df_train': df_train, 'df_val': df_val, 'df_test': df_test}
