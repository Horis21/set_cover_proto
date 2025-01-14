from dataclasses import dataclass, field
import pandas as pd


@dataclass(eq=True, frozen=True)
class HashableDataFrame:
    df: pd.DataFrame
    _hash: int = field(init=False, repr=False)

    def __post_init__(self):
        # Calculate the hash only once, when the object is initialized
        object.__setattr__(self, '_hash', hash(frozenset(self.df.index)))

    def __hash__(self):
        return self._hash
    
    def __getattr__(self, attr):
        # Delegate attribute access to the pandas DataFrame
        return getattr(self.df, attr)
    
    def __getitem__(self, key):
        """
        Delegate indexing (e.g., hdf[0], hdf['column_name']) to the underlying DataFrame.
        """
        return self.df[key]

    def __setitem__(self, key, value):
        """
        Delegate item assignment (e.g., hdf['new_column'] = values) to the underlying DataFrame.
        """
        self.df[key] = value

    def __len__(self):
        """
        Delegate len(hdf) to the underlying DataFrame.
        """
        return len(self.df)

    def __iter__(self):
        """
        Delegate iteration (e.g., for col in hdf) to the underlying DataFrame.
        """
        return iter(self.df)
    
    def __eq__(self, other):
        if not isinstance(other, HashableDataFrame):
             raise TypeError('Can only compare two HashableDataFrames')
        return self.df.index.equals(other.df.index)
