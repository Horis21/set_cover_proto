def hashable(df):
        return frozenset((tuple(x) for x in df.values))

class Cache:
    def __init__(self):
        self.upper = {}
        self.lower = {}
        self.solution = {}
        self.one_off = {}
        self.vertex_cover = {}
        self.possible_feats = {}

    def get_possbile_feats(self, df):
        return self.possible_feats.get(hashable(df))
    
    def put_possible_feats(self, df, feats):
        self.possible_feats[hashable(df)] = feats

    def get_one_offs(self, df):
        return self.one_off.get(hashable(df), {})
    
    def put_one_offs(self, df, one_offs):
         self.one_off[hashable(df)] = one_offs

    def get_vertex_cover(self, df):
        return self.vertex_cover.get(hashable(df), {})
    
    def put_vertex_cover(self, df, vertex_cover):
         self.vertex_cover[hashable(df)] = vertex_cover

    def get_solution(self, df):
        return self.solution.get(hashable(df))
    
    def put_solution(self, df, solution):
         solution.feasible = True
         self.solution[hashable(df)] = solution

    def get_lower(self, df):
        return self.lower.get(hashable(df))
    
    def put_lower(self, df, bound):
         lower = self.get_lower(df)
         if lower is None or lower < bound:
              self.lower[hashable(df)] = bound

    def get_upper(self, df):
        return self.upper.get(hashable(df))
    
    def put_upper(self, df, bound):
         upper = self.get_upper(df)
         if upper is None or upper > bound:
              self.upper[hashable(df)] = bound
