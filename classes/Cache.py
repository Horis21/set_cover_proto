from typing import OrderedDict
from classes.CacheEntry import CacheEntry


class Cache:
    def __init__(self):
        self.max_size = 100000  # Maximum size of the cache
        self.entries = OrderedDict()  # Maintains insertion order for LRU

    def get_entry(self, df) -> CacheEntry:
        """
        Retrieves the CacheEntry for the given DataFrame.
        If it doesn't exist, creates a new entry.
        """
        # Move the accessed entry to the end (most recently used)
        if df in self.entries:
            self.entries.move_to_end(df)
        else:
            # Create a new CacheEntry

            # Remove the most recently used entry if the cache exceeds max size
            if len(self.entries) > self.max_size:
                self.entries.popitem(last=False)  # Remove the last item (LRU)

            self.entries[df] = CacheEntry()

        return self.entries[df]

    def __contains__(self, df) -> bool:
        """Checks if the DataFrame is in the cache."""
        return df in self.entries

    def __len__(self) -> int:
        """Returns the number of entries in the cache."""
        return len(self.entries)