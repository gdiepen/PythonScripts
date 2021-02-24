"""Module for holding a onehotencoder that can easily be applied to a complete
dataframe that also sets the feature/column names correctly

Author: Guido Diepen
License: MIT
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np


class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    """Specialized version of OneHotEncoder that plays nice with pandas DataFrames and
    will automatically set the feature/column names after fit/transform
    """

    def __init__(
        self,
        categories="auto",
        drop=None,
        sparse=None,
        dtype=np.float64,
        handle_unknown="error",
        col_overrule_params={},
    ):
        """Create DataFrameOneHotEncoder that can be fitted to and transform dataframes
        and that will set up the column/feature names automatically to
        original_column_name[categorical_value]

        If you provide the same arguments as you would for the sklearn 
        OneHotEncoder, these parameters will apply for all of the columns. If you want
        to have specific overrides for some of the columns, provide these in the dict
        argument col_overrule_params.
        
        For example:
            DataFrameOneHotEncoder(col_overrule_params={"col2":{"drop":"first"}})

        will create a OneHotEncoder for each of the columns with default values, but
        uses a drop=first argument for columns with the name col2

        Args:
            categories‘auto’ or a list of array-like, default=’auto’
                ‘auto’ : Determine categories automatically from the training data.
                list : categories[i] holds the categories expected in the ith column.
                The passed categories should not mix strings and numeric values
                within a single feature, and should be sorted in case of numeric
                values.
            drop: {‘first’, ‘if_binary’} or a array-like of shape (n_features,),
                default=None
                See OneHotEncoder documentation
            sparse: Ignored, since we always will work with dense dataframes
            dtype: number type, default=float
                Desired dtype of output.
            handle_unknown: {‘error’, ‘ignore’}, default=’error’
                Whether to raise an error or ignore if an unknown categorical feature
                is present during transform (default is to raise). When this parameter
                is set to ‘ignore’ and an unknown category is encountered during
                transform, the resulting one-hot encoded columns for this feature will
                be all zeros. In the inverse transform, an unknown category will be
                denoted as None.
            col_overrule_params: dict of {column_name: dict_params} where dict_params
                are exactly the options cateogires,drop,sparse,dtype,handle_unknown.
                For the column given by the key, these values will overrule the default
                parameters
        """
        self.categories = categories
        self.drop = drop
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.col_overrule_params = col_overrule_params
        pass

    def fit(self, X, y=None):
        """Fit a separate OneHotEncoder for each of the columns in the dataframe

        Args:
            X: dataframe
            y: None, ignored. This parameter exists only for compatibility with
                Pipeline

        Returns
            self

        Raises
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        self.onehotencoders_ = []
        self.column_names_ = []

        for c in X.columns:
            # Construct the OHE parameters using the arguments
            ohe_params = {
                "categories": self.categories,
                "drop": self.drop,
                "sparse": False,
                "dtype": self.dtype,
                "handle_unknown": self.handle_unknown,
            }
            # and update it with potential overrule parameters for the current column
            ohe_params.update(self.col_overrule_params.get(c, {}))

            # Regardless of how we got the parameters, make sure we always set the
            # sparsity to False
            ohe_params["sparse"] = False

            # Now create, fit, and store the onehotencoder for current column c
            ohe = OneHotEncoder(**ohe_params)
            self.onehotencoders_.append(ohe.fit(X.loc[:, [c]]))

            # Get the feature names and replace each x0_ with empty and after that
            # surround the categorical value with [] and prefix it with the original
            # column name
            feature_names = ohe.get_feature_names()
            feature_names = [x.replace("x0_", "") for x in feature_names]
            feature_names = [f"{c}[{x}]" for x in feature_names]

            self.column_names_.append(feature_names)

        return self

    def transform(self, X):
        """Transform X using the one-hot-encoding per column

        Args:
            X: Dataframe that is to be one hot encoded

        Returns:
            Dataframe with onehotencoded data

        Raises
            NotFittedError if the transformer is not yet fitted
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        if not hasattr(self, "onehotencoders_"):
            raise NotFittedError(f"{type(self).__name__} is not fitted")

        all_df = []

        for i, c in enumerate(X.columns):
            ohe = self.onehotencoders_[i]

            transformed_col = ohe.transform(X.loc[:, [c]])

            df_col = pd.DataFrame(transformed_col, columns=self.column_names_[i])
            all_df.append(df_col)

        return pd.concat(all_df, axis=1)
