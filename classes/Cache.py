from dataclasses import dataclass, field
import pandas as pd

@dataclass(eq=True, frozen=True)
class HashableDataFrame:
    df: pd.DataFrame
    _hash: int = field(init=False, repr=False)

    def __post_init__(self):
        # Calculate the hash only once, when the object is initialized
        object.__setattr__(self, '_hash', hash(frozenset(tuple(x) for x in self.df.values)))

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

class Cache:
    def __init__(self):
        self.upper = {}
        self.lower = {}
        self.solution = {}
        self.one_off = {}
        self.vertex_cover = {}
        self.possible_feats = {}
        self.bests = {}
        self.dts = {}

    def put_dt(self, df, dt):
        self.dts[hash(df)] = dt

    def get_dt(self, df):
        return self.dts[hash(df)]

    def get_best(self, df):
        return self.bests.get(hash(df))

    def get_possbile_feats(self, df):
        return self.possible_feats.get(hash(df))
    
    def put_possible_feats(self, df, feats):
        self.possible_feats[hash(df)] = feats

    def get_one_offs(self, df):
        return self.one_off.get(hash(df))
    
    def put_best(self, df, best):
         self.bests[hash(df)] = best
    
    def put_one_offs(self, df, one_offs):
         self.one_off[hash(df)] = one_offs

    def get_vertex_cover(self, df):
        return self.vertex_cover.get(hash(df))
    
    def put_vertex_cover(self, df, vertex_cover):
         self.vertex_cover[hash(df)] = vertex_cover

    def get_solution(self, df):
        return self.solution.get(hash(df))
    
    def put_solution(self, df, solution):
         solution.feasible = True
         self.solution[hash(df)] = solution

    def get_lower(self, df):
        return self.lower.get(hash(df))
    
    def put_lower(self, df, bound):
         lower = self.get_lower(df)
         if lower is None or lower < bound:
              self.lower[hash(df)] = bound

    def get_upper(self, df):
        return self.upper.get(hash(df))
    
    def put_upper(self, df, bound):
        upper = self.get_upper(df)
        if upper is None or upper > bound:
              self.upper[hash(df)] = bound
              return True
        return False

