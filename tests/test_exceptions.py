import unittest

import pandas as pd
import pytest

from source.exception import StratifiedClassHasOneMember
from source.splitter import splitter


class TestExceptions(unittest.TestCase):
    """
    Tests to check right handling of Exceptions
    """
    def test_df_src_is_empty(self):
        """
        If we pass empty DataFrame
        We expect Exception
        """
        df_src = pd.DataFrame()
        with pytest.raises(ValueError):
            splitter(df_src, 0.1)

    def test_train_test_size(self):
        """
        If we pass bad train_size or test_size
        We expect Exception
        """
        df_src = pd.DataFrame({'a': [1, 2, 3]})  # simple df_src for testing
        train_size1 = 1.1
        train_size2 = -0.0001

        with pytest.raises(ValueError):
            splitter(df_src, train_size1)

        with pytest.raises(ValueError):
            splitter(df_src, train_size2)

        test_size1 = 1.1
        test_size2 = -0.0001

        with pytest.raises(ValueError):
            splitter(df_src, 0.2, test_size=test_size1)

        with pytest.raises(ValueError):
            splitter(df_src, 0.2, test_size=test_size2)

    def test_train_test_size_sum(self):
        """
        We pass train_size and test_size with corresponding sum > 1
        We expect Exception
        """
        df_src = pd.DataFrame({'a': [1, 2, 3]})  # simple df_src for testing
        train_size = 0.9
        test_size = 0.2  # sum is bigger than 1

        with pytest.raises(ValueError):
            splitter(df_src, train_size, test_size=test_size)

    def test_stratify_train_test(self):
        """
        We split our df_src with only 1 member of class '1'
        We expect Exception from sklearn library
        """
        df_src = pd.DataFrame({'a': [1, 2, 3], 'y': [0, 0, 1]})
        stratification_column = 'y'

        with pytest.raises(StratifiedClassHasOneMember):
            splitter(
                df_src,
                train_size=0.5,
                test_size=0.5,
                split_method='random_with_stratification',
                stratification_column=stratification_column,
                seed=0
            )

    def test_stratify_train_test_valid(self):
        """
        We split df_src to df_train and val_and_test_df
        So val_and_test_df['y'] = [0, 1], only 1 member for each class
        We expect Exception from sklearn library
        """
        df_src = pd.DataFrame({'a': [1, 2, 3, 4], 'y': [0, 0, 1, 1]})
        stratification_column = 'y'

        with pytest.raises(Exception):
            splitter(
                df_src,
                train_size=0.5,
                test_size=0.25,
                split_method='random_with_stratification',
                stratification_column=stratification_column,
                seed=0
            )
