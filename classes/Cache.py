def hashable(df):
        return frozenset((tuple(x) for x in df.values))

class Cache:
    def __init__(self):
        self.upper = {}
        self.lower = {}
        self.solution = {}
        self.one_off = {}
        self.vertex_cover = {}


    def get_one_offs(self, df):
        return self.one_off[hashable(df)]
    
    def put_one_offs(self, df, one_offs):
         self.one_off[hashable(df)] = one_offs

    def get_vertex_cover(self, df):
        return self.vertex_cover[hashable(df)]
    
    def put_vertex_cover(self, df, vertex_cover):
         self.vertex_cover[hashable(df)] = vertex_cover

    def get_solution(self, df):
        return self.solution[hashable(df)]
    
    def put_solution(self, df, solution):
         self.solution[hashable(df)] = solution

    def get_lower(self, df):
        return self.lower[hashable(df)]
    
    def put_lower(self, df, bound):
         self.lower[hashable(df)] = bound

    def get_upper(self, df):
        return self.upper[hashable(df)]
    
    def put_upper(self, df, bound):
         self.upper[hashable(df)] = bound
