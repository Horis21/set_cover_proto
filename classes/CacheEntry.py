class CacheEntry:
    def __init__(self):
        self.upper = None
        self.lower = None
        self.solution = None
        self.one_off = None
        self.set_cover = None
        self.possible_feats = None
        self.best = None
        self.dt = None

    def get_dt(self):
        return self.dt

    def get_best(self):
        return self.best

    def get_possbile_feats(self):
        return self.possible_feats

    def get_one_offs(self):
        return self.one_off

    def get_set_cover(self):
        return self.set_cover

    def get_solution(self):
        return self.solution

    def get_lower(self):
        return self.lower

    def get_upper(self):
        return self.upper

    def put_upper(self, bound):
        if self.upper is None or self.upper > bound:
                self.upper = bound
                return True
        return False

    def put_lower(self, bound):
        if self.lower is None or self.lower < bound:
            self.lower = bound

    def put_set_cover(self, set_cover):
            self.set_cover = set_cover

    def put_solution(self, solution):
        self.solution = solution

    def put_one_offs(self, one_offs):
        self.one_off = one_offs

    def put_possible_feats(self, feats):
        self.possible_feats = feats

    def put_best(self, best):
        self.best = best
        
    def put_dt(self, dt):
        self.dt = dt