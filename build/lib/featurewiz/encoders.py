###############################################################################
# MIT License
#
# Copyright (c) 2020 Alex Lekov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
#####  This amazing Library was created by Alex Lekov: Many Thanks to Alex! ###
#####                https://github.com/Alex-Lekov/AutoML_Alex              ###
###############################################################################
import pandas as pd
import numpy as np

################################################################
            #               Simple Encoders
            #      (do not use information about target)
################################################################

class FrequencyEncoder():
    """
    FrequencyEncoder
    Conversion of category into frequencies.
    Parameters
        ----------
    cols : list of categorical features.
    drop_invariant : not used
    """
    def __init__(self, cols=None, drop_invariant=None):
        """
        Description of __init__

        Args:
            cols=None (undefined): columns in dataset
            drop_invariant=None (undefined): not used

        """
        self.cols = cols
        self.counts_dict = None

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Description of fit

        Args:
            X (pd.DataFrame): dataset
            y=None (not used): not used

        Returns:
            pd.DataFrame

        """
        counts_dict = {}
        if self.cols is None:
            self.cols = X.columns
        for col in self.cols:
            values = X[col].value_counts(dropna=False).index
            n_obs = np.float(len(X))
            counts = list(X[col].value_counts(dropna=False) / n_obs)
            counts_dict[col] = dict(zip(values, counts))
        self.counts_dict = counts_dict

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Description of transform

        Args:
            X (pd.DataFrame): dataset

        Returns:
            pd.DataFrame

        """
        counts_dict_test = {}
        res = []
        for col in self.cols:
            values = X[col].value_counts(1,dropna=False).index.tolist()
            counts = X[col].value_counts(1,dropna=False).values.tolist()
            counts_dict_test[col] = dict(zip(values, counts))

            # if value is in "train" keys - replace "test" counts with "train" counts
            for k in [
                key
                for key in counts_dict_test[col].keys()
                if key in self.counts_dict[col].keys()
            ]:
                counts_dict_test[col][k] = self.counts_dict[col][k]
            res.append(X[col].map(counts_dict_test[col]).values.reshape(-1, 1))
        try:
            res = np.hstack(res)
        except:
            pdb.set_trace()
        X[self.cols] = res
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Description of fit_transform

        Args:
            X (pd.DataFrame): dataset
            y=None (undefined): not used

        Returns:
            pd.DataFrame

        """
        self.fit(X, y)
        X = self.transform(X)
        return X
