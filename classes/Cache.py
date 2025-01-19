from classes.CacheEntry import CacheEntry


class Cache:
    def __init__(self):
        self.entries = {}

    def get_entry(self, df) -> CacheEntry:
        if self.entries.get(df) is None:
            self.entries[df] = CacheEntry()
        return self.entries[df]
    