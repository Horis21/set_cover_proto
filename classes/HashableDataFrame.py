from dataclasses import dataclass, field
import pandas as pd


class HashableDataFrame:

    def __init__(self, df, indices=None):
        """
        A wrapper for a DataFrame that allows operations on subsets based on row indices.
        """
        self.df = df
        self._hash = hash(frozenset(indices if indices is not None else df.index))
        self.indices = indices if indices is not None else df.index

    def subset(self, condition):
        """
        Returns a new HashableDataFrame for rows that satisfy the condition.
        Args:
            condition (Series): A boolean Series indicating the rows to include.
        """
        new_indices = self.indices[condition]
        return HashableDataFrame(self.get_df(), new_indices)

    def get_rows(self):
        """
        Returns the DataFrame rows corresponding to the stored indices.
        """
        return self.df.loc[self.indices]

    def __hash__(self):
        return self._hash
    
    def get_indices(self):
        return self.indices
    
    def get_df(self):
        return self.df
    
    def __getattr__(self, attr):
        # Delegate attribute access to the pandas DataFrame
        return getattr(self.get_rows(), attr)
    
    def __getitem__(self, key):
        """
        Delegate indexing (e.g., hdf[0], hdf['column_name']) to the underlying DataFrame.
        """
        return self.get_rows()[key]


    def __len__(self):
        """
        Delegate len(hdf) to the underlying DataFrame.
        """
        return len(self.get_rows())

    def __iter__(self):
        """
        Delegate iteration (e.g., for col in hdf) to the underlying DataFrame.
        """
        return iter(self.get_rows())
    
    def __eq__(self, other):
        if not isinstance(other, HashableDataFrame):
             raise TypeError('Can only compare two HashableDataFrames')
        return self.indices.equals(other.indices)
