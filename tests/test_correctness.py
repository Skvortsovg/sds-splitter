import unittest

import numpy as np
import pandas as pd

from source.splitter import splitter


# TODO: doc
class TestCorrectness(unittest.TestCase):
    def setUp(self):
        self.size = 1000
        self.y = [i % 4 + 1 for i in range(self.size)]
        self.x = [i for i in range(self.size)]

        np.random.seed(0)
        np.random.shuffle(self.y)

        self.df_src = pd.DataFrame({'x': self.x, 'y': self.y})

    def test_stratification(self):
        # Without validation DataFrame
        split_result = splitter(
            self.df_src,
            0.5,
            stratification_column='y',
            split_method='random_with_stratification'
        )
        df_train, df_test = split_result['df_train'], split_result['df_test']

        self.assertEqual(df_train.y.sum(), df_test.y.sum())

        # With validation DataFrame
        split_result = splitter(
            self.df_src,
            0.5,
            test_size=0.25,
            stratification_column='y',
            split_method='random_with_stratification'
        )
        df_train, df_test, df_val = split_result['df_train'], \
                                    split_result['df_test'], \
                                    split_result['df_val'],

        self.assertLess(abs(df_test.y.sum()-df_val.y.sum()), 2)

    def test_no_stratification(self):
        split_result = splitter(self.df_src, 0.5)
        df_train, df_test = split_result['df_train'], split_result['df_test']

        self.assertNotEqual(df_train.y.sum(), df_test.y.sum())

    def test_no_val(self):
        split_result = splitter(self.df_src, 0.5)
        df_val = split_result['df_val']

        self.assertEqual(len(df_val), 0)

    def test_int_train_test_conv(self):
        train_size = 100
        test_size = 300

        split_result = splitter(self.df_src, train_size, test_size=test_size)
        df_train, df_test = split_result['df_train'], split_result['df_test']

        self.assertEqual(len(df_train), train_size)
        self.assertEqual(len(df_test), test_size)

    def test_output_shapes(self):
        train_size = 0.5
        test_size = 0.3

        split_result = splitter(self.df_src, train_size, test_size=test_size)
        df_train, df_test, df_val = split_result['df_train'], \
                                    split_result['df_test'], \
                                    split_result['df_val']

        self.assertEqual(df_train.shape[0], self.size * train_size)
        self.assertEqual(df_test.shape[0], self.size * test_size)
        self.assertEqual(df_val.shape[0], self.size * (1 - train_size - test_size))
